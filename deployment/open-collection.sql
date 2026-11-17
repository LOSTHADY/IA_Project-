begin;

alter table public.reading_invites
  add column if not exists participant_name text
  check (participant_name is null or char_length(participant_name) between 1 and 60);

create or replace function public.open_readings(
  display_name text default null,
  accept_research boolean default false,
  accept_public boolean default false,
  invitation uuid default null
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
    insert into public.reading_invites (participant_name) values (participant_label)
      returning token into participation_token;
  end if;

  -- A resume uses the consent recorded for the existing session. New sessions
  -- reach this point only through an explicit acceptance in the form.
  result := public.start_readings(participation_token, true, accept_public, 'lecture-v1');
  if display_name is not null then
    update public.reading_invites set participant_name = participant_label
    where token = participation_token and participant_name is null;
  end if;
  return result;
end;
$$;

revoke all on function public.open_readings(text, boolean, boolean, uuid) from public, anon, authenticated;
grant execute on function public.open_readings(text, boolean, boolean, uuid) to authenticated;

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
           c.duration_ms, p.sentence_bm,
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

commit;
