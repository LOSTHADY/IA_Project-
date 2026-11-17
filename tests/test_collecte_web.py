"""Collecte web : outils (phrases, import) et fonctions SQL de Supabase.

Les fonctions SQL tournent sur un vrai PostgreSQL quand `COLLECTE_PG_DSN` est
défini (la CI en démarre un), avec des imitations des schémas `auth` et
`storage` de Supabase. Sinon, ces tests sont sautés.
"""

from __future__ import annotations

import io
import json
import os
import sys
import uuid
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collecte_web  # noqa: E402


def test_phrases_sql_echappe_les_apostrophes():
    sql = collecte_web.phrases_sql([{"bm": "A ka ɲi", "fr": "C'est bien"}])
    assert "insert into public.reading_prompts" in sql
    assert "'C''est bien'" in sql and "'A ka ɲi'" in sql
    assert collecte_web.phrases_sql([]).startswith("--")


def _wav(secondes: float = 1.0) -> bytes:
    t = np.arange(int(16000 * secondes)) / 16000
    s = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(s.tobytes())
    return buf.getvalue()


def _export(chemin: Path, lignes: list[dict]) -> Path:
    with zipfile.ZipFile(chemin, "w") as z:
        for r in lignes:
            z.writestr(r["audio"], _wav())
        z.writestr("testset.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lignes))
    return chemin


def test_importer_dans_l_atelier(tmp_path):
    lignes = [{"id": str(uuid.uuid4()), "audio": f"audio/{k}.wav", "speaker": spk, "gender": g,
               "prompt_id": p, "register": "lu", "transcript_bm": f"I ni sɔgɔma {p}",
               "translation_fr": "Bonjour", "code_switching": False, "duration_s": 1.0}
              for k, (spk, g, p) in enumerate([("spk-web-a", "f", 1), ("spk-web-a", "f", 2),
                                                ("spk-web-b", "", 1)])]
    z = _export(tmp_path / "export.zip", lignes)
    atelier = tmp_path / "atelier"
    assert collecte_web.importer(z, atelier) == {"lectures": 3, "lecteurs": 2, "deja_importees": 0}
    assert collecte_web.importer(z, atelier) == {"lectures": 0, "lecteurs": 0, "deja_importees": 3}
    prises = [json.loads(l) for l in (atelier / "lire.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {p["prompt_id"] for p in prises} == {"web-1", "web-2"}
    assert all((atelier / p["audio"]).exists() for p in prises)
    from affiner_soloni import exemples
    ex = exemples(atelier)
    assert len(ex) == 3 and all(e["source"] == "lecture" for e in ex)
    # une même phrase tombe du même côté pour tous ses lecteurs
    cote = {e["text"]: e["partie"] for e in ex if e["speaker"] == "spk-web-a"}
    assert all(cote[e["text"]] == e["partie"] for e in ex if e["text"] in cote)


# --- fonctions SQL (Supabase imité) --------------------------------------------

DSN = os.environ.get("COLLECTE_PG_DSN")
IMITATION = """
do $$ begin create role anon; exception when duplicate_object then null; end $$;
do $$ begin create role authenticated; exception when duplicate_object then null; end $$;
create schema auth; create table auth.users(id uuid primary key);
create function auth.uid() returns uuid language sql stable as
  $f$ select nullif(current_setting('test.uid', true), '')::uuid $f$;
create function auth.jwt() returns jsonb language sql stable as
  $f$ select coalesce(nullif(current_setting('test.jwt', true), ''), '{}')::jsonb $f$;
create schema storage;
create table storage.buckets(id text primary key, name text, public boolean,
  file_size_limit bigint, allowed_mime_types text[]);
create table storage.objects(id uuid default gen_random_uuid(), bucket_id text, name text);
alter table storage.objects enable row level security;
"""


@pytest.fixture
def base():
    if not DSN:
        pytest.skip("COLLECTE_PG_DSN non défini : pas de PostgreSQL pour les fonctions SQL")
    psycopg = pytest.importorskip("psycopg")
    c = psycopg.connect(DSN, autocommit=True)
    c.execute("drop schema if exists public cascade; create schema public;"
              "drop schema if exists auth cascade; drop schema if exists storage cascade;")
    c.execute(IMITATION)
    for f in ("collection.sql", "open-collection.sql", "collecte-v2.sql"):
        c.execute((ROOT / "deployment" / f).read_text(encoding="utf-8"))
    c.execute((ROOT / "deployment" / "collecte-v2.sql").read_text(encoding="utf-8"))  # rejouable
    for k in range(25):
        c.execute("insert into public.reading_prompts(sentence_bm, translation_fr) values (%s, %s)",
                  (f"phrase {k}", f"fr {k}"))
    yield c
    c.close()


def _qui(c, uid, anonyme=True):
    c.execute("select set_config('test.uid', %s, false), set_config('test.jwt', %s, false)",
              (str(uid), json.dumps({"is_anonymous": anonyme})))


def _un(c, sql, *args):
    return c.execute(sql, args).fetchone()[0]


def test_sql_parcours_complet(base):
    import psycopg
    c = base
    a, b, t1, t2 = (uuid.uuid4() for _ in range(4))
    for u in (a, b, t1, t2):
        c.execute("insert into auth.users values (%s)", (u,))
    _qui(c, a)
    pa = _un(c, "select public.open_readings('Awa', true, false, null, 'f')")
    _qui(c, b)
    pb = _un(c, "select public.open_readings('Moussa', true, false, null, 'm')")
    ia, ib = {p["id"] for p in pa["prompts"]}, {p["id"] for p in pb["prompts"]}
    assert len(ia) == len(ib) == 10 and not ia & ib  # les moins lues d'abord

    _qui(c, a)
    assert len(_un(c, "select public.more_readings()")["prompts"]) == 10  # il en reste à lire
    for p in pa["prompts"]:
        clip = _un(c, "select public.reserve_reading(%s, 3000)", p["id"])
        c.execute("insert into storage.objects(bucket_id, name) values ('bambara-readings', %s)",
                  (clip["id"] + ".wav",))
        assert _un(c, "select public.finish_reading(%s)", clip["id"])
    plus = _un(c, "select public.more_readings()")
    nouvelles = {p["id"] for p in plus["prompts"] if not p["done"]}
    assert len(plus["prompts"]) == 20 and len(nouvelles) == 10 and len(nouvelles & ib) == 5
    assert len(_un(c, "select public.open_readings()")["prompts"]) == 20  # reprise

    c.execute("insert into public.reading_team values (%s), (%s)", (t1, t2))
    _qui(c, t1, False)
    lignes = _un(c, "select public.list_readings()")
    assert len(lignes) == 10 and lignes[0]["gender"] == "f" and lignes[0]["prompt_id"] in ia
    c.execute("select public.review_reading(%s, 'transcribe', 'phrase', 'fr', false)", (lignes[0]["id"],))
    with pytest.raises(psycopg.errors.InvalidParameterValue):  # pas de validation par soi-même
        c.execute("select public.review_reading(%s, 'approve', 'phrase', 'fr', false)", (lignes[0]["id"],))
    _qui(c, t2, False)
    c.execute("select public.review_reading(%s, 'approve', 'phrase', 'fr', false)", (lignes[0]["id"],))
    validees = _un(c, "select public.list_readings(null, true, false)")
    assert len(validees) == 1 and validees[0]["participant_name"] is None

    _qui(c, uuid.uuid4())
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        c.execute("select public.open_readings('X', true, false, null, 'z')")
