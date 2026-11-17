-- Collecte web, version 2. À exécuter une fois dans le SQL Editor de Supabase,
-- après collection.sql puis open-collection.sql.
--
-- 1. Chaque participant reçoit les phrases les moins lues (au hasard entre
--    elles), au lieu des dix mêmes pour tout le monde : la reconnaissance
--    apprend la langue, pas dix phrases.
-- 2. « 10 phrases de plus » : une fois ses phrases lues, on peut continuer
--    (jusqu'à 200 phrases par participation).
-- 3. Le genre, facultatif, pour suivre la parité visée par la campagne.
-- 4. L'export donne le numéro de la phrase et le genre.
begin;

alter table public.reading_invites
  add column if not exists gender text check (gender is null or gender in ('f', 'm'));

-- Les phrases actives les moins attribuées d'abord, au hasard entre elles.
create or replace function public.least_read_prompts(excluded integer[], how_many integer)
returns integer[]
language sql volatile security definer set search_path = ''
as $$
  select coalesce(array_agg(id), '{}') from (
    select p.id from public.reading_prompts p
    where p.active and not (p.id = any(coalesce(excluded, '{}')))
    order by (select count(*) from public.reading_invites i where p.id = any(i.prompt_ids)), random()
    limit how_many
  ) choisies;
$$;

create or replace function public.start_readings(invitation uuid, accept_research boolean,
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
    assigned := public.least_read_prompts('{}', 10);
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
        'done', c.submitted_at is not null) order by array_position(session_row.prompt_ids, p.id))
      from public.reading_prompts p
      left join public.reading_clips c on c.prompt = p.id and c.invite = invitation
      where p.id = any(session_row.prompt_ids))
  );
end;
$$;

create or replace function public.more_readings() returns jsonb
language plpgsql security definer set search_path = ''
as $$
declare
  participation public.reading_invites%rowtype;
  extra integer[];
begin
  select * into participation from public.reading_invites
  where user_id = auth.uid() and consent_at is not null and not revoked and expires_at > now()
  for update;
  if not found then
    raise exception 'Cette participation est terminée sur cet appareil.' using errcode = '22023';
  end if;
  -- des phrases restent à lire : on ne rajoute rien
  if exists (
    select 1 from unnest(participation.prompt_ids) as attribuee(id)
    where not exists (select 1 from public.reading_clips c
                      where c.invite = participation.token and c.prompt = attribuee.id
                        and c.submitted_at is not null)
  ) then
    return public.start_readings(participation.token, true, participation.public_ok, 'lecture-v1');
  end if;
  if cardinality(participation.prompt_ids) >= 200 then
    raise exception 'Merci ! Vous avez lu le maximum de phrases pour cette participation.'
      using errcode = '22023';
  end if;
  extra := public.least_read_prompts(participation.prompt_ids, 10);
  if cardinality(extra) = 0 then
    raise exception 'Vous avez lu toutes les phrases disponibles. Merci !' using errcode = '22023';
  end if;
  update public.reading_invites set prompt_ids = prompt_ids || extra
  where token = participation.token;
  return public.start_readings(participation.token, true, participation.public_ok, 'lecture-v1');
end;
$$;

drop function if exists public.open_readings(text, boolean, boolean, uuid);
drop function if exists public.open_readings(text, boolean, boolean, uuid, text);

create function public.open_readings(
  display_name text default null,
  accept_research boolean default false,
  accept_public boolean default false,
  invitation uuid default null,
  participant_gender text default null
) returns jsonb
language plpgsql security definer set search_path = ''
as $$
declare
  participant uuid := auth.uid();
  participant_label text := btrim(display_name);
  participation public.reading_invites%rowtype;
  participation_token uuid := invitation;
  result jsonb;
begin
  if participant is null or not coalesce((auth.jwt()->>'is_anonymous')::boolean, false) then
    raise exception 'Session de participation requise.' using errcode = '42501';
  end if;

  if display_name is not null and (
    accept_research is distinct from true
    or char_length(participant_label) not between 1 and 60
    or participant_label ~ '[[:cntrl:]]'
  ) then
    raise exception 'Indiquez un prénom ou un pseudo et acceptez de participer.' using errcode = '22023';
  end if;
  if participant_gender is not null and participant_gender not in ('f', 'm') then
    raise exception 'Genre inconnu.' using errcode = '22023';
  end if;

  perform pg_advisory_xact_lock(hashtextextended('bambara-readings:' || participant::text, 0));

  select * into participation from public.reading_invites where user_id = participant;
  if found then
    if participation.revoked or participation.expires_at <= now() or participation.consent_at is null then
      if display_name is null then return null; end if;
      raise exception 'Cette participation est terminée sur cet appareil.' using errcode = '22023';
    end if;
    if invitation is not null and invitation <> participation.token then
      raise exception 'Une participation est déjà ouverte dans ce navigateur.' using errcode = '22023';
    end if;
    participation_token := participation.token;
  elsif display_name is null then
    return null;
  elsif participation_token is null then
    insert into public.reading_invites (participant_name, gender) values (participant_label, participant_gender)
      returning token into participation_token;
  end if;

  -- Une reprise garde l'accord déjà enregistré. Une nouvelle participation
  -- n'arrive ici qu'après l'acceptation explicite du formulaire.
  result := public.start_readings(participation_token, true, accept_public, 'lecture-v1');
  if display_name is not null then
    update public.reading_invites
    set participant_name = coalesce(participant_name, participant_label),
        gender = coalesce(gender, participant_gender)
    where token = participation_token;
  end if;
  return result;
end;
$$;

create or replace function public.list_readings(after_id uuid default null, approved_only boolean default false,
  public_only boolean default false) returns jsonb
language plpgsql stable security definer set search_path = ''
as $$
begin
  if not public.is_reading_team() then
    raise exception 'Accès réservé à l’équipe.' using errcode = '42501';
  end if;
  return (select coalesce(jsonb_agg(to_jsonb(rows) order by rows.id), '[]'::jsonb)
  from (
    select c.id, i.speaker, case when not approved_only and not public_only then i.participant_name end as participant_name,
           i.gender, c.prompt as prompt_id, c.duration_ms, p.sentence_bm,
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

revoke all on function public.least_read_prompts(integer[], integer) from public, anon, authenticated;
revoke all on function public.more_readings(),
  public.open_readings(text, boolean, boolean, uuid, text) from public, anon, authenticated;
grant execute on function public.more_readings(),
  public.open_readings(text, boolean, boolean, uuid, text) to authenticated;

commit;
