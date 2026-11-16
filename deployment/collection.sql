begin;

create table public.reading_prompts (
  id integer generated always as identity primary key,
  sentence_bm text not null check (length(trim(sentence_bm)) between 1 and 500),
  translation_fr text not null check (length(trim(translation_fr)) between 1 and 500),
  active boolean not null default true
);

create table public.reading_invites (
  token uuid primary key default gen_random_uuid(),
  speaker text not null unique default ('spk-web-' || gen_random_uuid()::text),
  expires_at timestamptz not null default now() + interval '14 days',
  revoked boolean not null default false,
  user_id uuid unique references auth.users(id),
  consent_at timestamptz,
  consent_version text,
  public_ok boolean not null default false,
  prompt_ids integer[]
);

create table public.reading_team (
  user_id uuid primary key references auth.users(id) on delete cascade
);

create table public.reading_clips (
  id uuid primary key default gen_random_uuid(),
  invite uuid not null references public.reading_invites(token),
  prompt integer not null references public.reading_prompts(id),
  duration_ms integer not null check (duration_ms between 500 and 20000),
  submitted_at timestamptz,
  transcript_bm text,
  translation_fr text,
  transcriber uuid references public.reading_team(user_id),
  reviewer uuid references public.reading_team(user_id),
  verdict text check (verdict in ('approved', 'rejected')),
  code_switching boolean not null default false,
  unique (invite, prompt),
  check (reviewer is null or (transcriber is not null and reviewer <> transcriber))
);

alter table public.reading_prompts enable row level security;
alter table public.reading_invites enable row level security;
alter table public.reading_team enable row level security;
alter table public.reading_clips enable row level security;
revoke all on public.reading_prompts, public.reading_invites,
  public.reading_team, public.reading_clips from anon, authenticated;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('bambara-readings', 'bambara-readings', false, 640044, array['audio/wav']);

create function public.may_upload_reading(object_name text) returns boolean
language sql stable security definer set search_path = ''
as $$
  select exists (
    select 1 from public.reading_clips c join public.reading_invites i on i.token = c.invite
    where c.id::text || '.wav' = object_name and c.submitted_at is null
      and i.user_id = auth.uid() and i.consent_at is not null
      and not i.revoked and i.expires_at > now()
  );
$$;

create function public.is_reading_team() returns boolean
language sql stable security definer set search_path = ''
as $$
  select exists (select 1 from public.reading_team t where t.user_id = auth.uid());
$$;

create policy reading_insert_reserved on storage.objects for insert to authenticated
with check (bucket_id = 'bambara-readings' and public.may_upload_reading(name));

create policy reading_select_team on storage.objects for select to authenticated
using (bucket_id = 'bambara-readings' and public.is_reading_team());

create function public.start_readings(invitation uuid, accept_research boolean,
  accept_public boolean, version text) returns jsonb
language plpgsql security definer set search_path = ''
as $$
declare
  session_row public.reading_invites%rowtype;
  assigned integer[];
begin
  if not coalesce((auth.jwt()->>'is_anonymous')::boolean, false)
     or accept_research is distinct from true or version is distinct from 'lecture-v1' then
    raise exception 'Accord de participation requis.' using errcode = '22023';
  end if;

  select * into session_row from public.reading_invites
  where token = invitation and not revoked and expires_at > now() for update;
  if not found or (session_row.user_id is not null and session_row.user_id <> auth.uid()) then
    raise exception 'Invitation invalide, expirée ou liée à un autre navigateur.' using errcode = '22023';
  end if;

  if session_row.user_id is null then
    assigned := array(
      select id from public.reading_prompts where active order by id limit 10
    );
    if cardinality(assigned) = 0 then
      raise exception 'Aucune phrase de lecture disponible.' using errcode = '22023';
    end if;
    update public.reading_invites
    set user_id = auth.uid(), consent_at = now(), consent_version = version,
        public_ok = coalesce(accept_public, false), prompt_ids = assigned
    where token = invitation returning * into session_row;
  end if;

  update public.reading_clips c set submitted_at = now()
  where c.invite = invitation and c.submitted_at is null
    and exists (select 1 from storage.objects o where o.bucket_id = 'bambara-readings'
                and o.name = c.id::text || '.wav');

  return jsonb_build_object(
    'speaker', session_row.speaker,
    'prompts', (select jsonb_agg(jsonb_build_object('id', p.id, 'text', p.sentence_bm,
        'done', c.submitted_at is not null) order by p.id)
      from public.reading_prompts p
      left join public.reading_clips c on c.prompt = p.id and c.invite = invitation
      where p.id = any(session_row.prompt_ids))
  );
end;
$$;

create function public.reserve_reading(prompt_id integer, milliseconds integer) returns jsonb
language plpgsql security definer set search_path = ''
as $$
declare
  invite_id uuid;
  clip_row public.reading_clips%rowtype;
begin
  select token into invite_id from public.reading_invites
  where user_id = auth.uid() and consent_at is not null and not revoked
    and expires_at > now() and prompt_id = any(prompt_ids) for update;
  if not found then
    raise exception 'Cette lecture ne fait pas partie de votre invitation.' using errcode = '22023';
  end if;
  insert into public.reading_clips (invite, prompt, duration_ms)
    values (invite_id, prompt_id, milliseconds) on conflict (invite, prompt)
    do nothing;
  select * into clip_row from public.reading_clips where invite = invite_id and prompt = prompt_id;
  if clip_row.submitted_at is null and not exists (
    select 1 from storage.objects where bucket_id = 'bambara-readings'
      and name = clip_row.id::text || '.wav'
  ) then
    update public.reading_clips set duration_ms = milliseconds where id = clip_row.id;
  end if;
  return jsonb_build_object('id', clip_row.id, 'done', clip_row.submitted_at is not null);
end;
$$;

create function public.finish_reading(clip_id uuid) returns boolean
language plpgsql security definer set search_path = ''
as $$
begin
  update public.reading_clips c set submitted_at = coalesce(c.submitted_at, now())
  from public.reading_invites i
  where c.id = clip_id and c.invite = i.token and i.user_id = auth.uid()
    and not i.revoked and i.expires_at > now() and exists (
      select 1 from storage.objects o where o.bucket_id = 'bambara-readings'
      and o.name = c.id::text || '.wav'
    );
  return found;
end;
$$;

create function public.list_readings(after_id uuid default null, approved_only boolean default false,
  public_only boolean default false) returns jsonb
language plpgsql stable security definer set search_path = ''
as $$
begin
  if not public.is_reading_team() then
    raise exception 'Accès réservé à l’équipe.' using errcode = '42501';
  end if;
  return (select coalesce(jsonb_agg(to_jsonb(rows) order by rows.id), '[]'::jsonb)
  from (
    select c.id, i.speaker, c.duration_ms, p.sentence_bm,
           coalesce(c.transcript_bm, p.sentence_bm) as transcript_bm,
           coalesce(c.translation_fr, p.translation_fr) as translation_fr,
           c.transcriber is not null as transcribed, c.transcriber = auth.uid() as mine,
           c.verdict, c.code_switching, i.public_ok
    from public.reading_clips c
    join public.reading_invites i on i.token = c.invite
    join public.reading_prompts p on p.id = c.prompt
    where c.submitted_at is not null and not i.revoked
      and (after_id is null or c.id > after_id)
      and (not approved_only or c.verdict = 'approved')
      and (not public_only or i.public_ok)
    order by c.id limit 30
  ) rows);
end;
$$;

create function public.review_reading(clip_id uuid, action text, bm text, fr text,
  switched boolean) returns void
language plpgsql security definer set search_path = ''
as $$
declare
  actor uuid := auth.uid();
begin
  if not public.is_reading_team() then
    raise exception 'Accès réservé à l’équipe.' using errcode = '42501';
  end if;
  if action is null or action not in ('transcribe', 'approve', 'reject')
    or length(trim(coalesce(bm, ''))) not between 1 and 1000
    or length(trim(coalesce(fr, ''))) not between 1 and 1000 then
    raise exception 'Transcription et traduction requises.' using errcode = '22023';
  end if;
  update public.reading_clips c
    set transcript_bm = trim(bm), translation_fr = trim(fr), code_switching = coalesce(switched, false),
      transcriber = case when action = 'transcribe' then actor else c.transcriber end,
      reviewer = case when action = 'transcribe' then null else actor end,
      verdict = case action when 'approve' then 'approved' when 'reject' then 'rejected' else null end
  from public.reading_invites i where c.invite = i.token and not i.revoked
    and c.id = clip_id and c.submitted_at is not null and c.verdict is null and (
    (action = 'transcribe' and (c.transcriber is null or c.transcriber = actor))
    or (action <> 'transcribe' and c.transcriber is not null and c.transcriber <> actor)
  );
  if not found then
    raise exception 'Cette action nécessite une deuxième personne après transcription.'
      using errcode = '22023';
  end if;
end;
$$;

revoke all on function public.may_upload_reading(text), public.is_reading_team(),
  public.start_readings(uuid, boolean, boolean, text), public.reserve_reading(integer, integer),
  public.finish_reading(uuid), public.list_readings(uuid, boolean, boolean),
  public.review_reading(uuid, text, text, text, boolean) from public, anon, authenticated;

grant execute on function public.may_upload_reading(text), public.is_reading_team(),
  public.start_readings(uuid, boolean, boolean, text), public.reserve_reading(integer, integer),
  public.finish_reading(uuid), public.list_readings(uuid, boolean, boolean),
  public.review_reading(uuid, text, text, text, boolean) to authenticated;

commit;
