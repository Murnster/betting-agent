# NFL props loop — setup guide for a new machine

This is the only thing the project runs in production: NFL player props, paper-traded
through the 2026 season, posted to Discord, graded the next morning. NBA/NHL/MLB code
exists in the repo but is frozen — nothing below touches it, and nothing below trains a
game model (the props models fit themselves from nflverse data on every run).

What a game day looks like once this is installed:

| When (local) | Job | What it does | Cost |
| --- | --- | --- | --- |
| 12:45 Fri/Sat/Sun, 19:00 Mon-Thu | `nfl_loop.sh card` | Fits the models, fetches today's slate, builds **one card per slate** (leans, TD scorer, props, straight overs, ladder hits), runs the Opus validator, saves picks, posts to Discord | 6 Odds API credits per game + 3 for the leans; ~$1 Opus per game |
| Hourly 09:00–23:00 daily | `nfl_loop.sh closing` | Closing price/line on held picks kicking off in the next 90 min → CLV | credits only for games with held picks |
| 09:00 daily | `nfl_loop.sh grade` | Finalizes games from nflverse, grades props, posts results + all-time recap | free |
| 03:15 daily | `nfl_loop.sh backup` | `pg_dump` to `backups/postgres/`, prunes >30 days | free |

Off-days cost nothing: `card` exits before any paid call when no game kicks off today,
and `closing` spends only when a held pick kicks off inside its window. That is why the
schedule can be blunt about days — one `card` run per day covers every slate that day
(`props.py --today` cards each slate in one pass and skips games already kicked off), so
the only thing that has to be right is running before the day's *first* kickoff. The
2026 season has games on all seven weekdays: a Wednesday opener (Sep 9) and a second
Wednesday in week 12, Friday games on Nov 27 and Christmas, December Saturdays.

---

## 1. Prerequisites

Install on the new laptop:

| Tool | Why | Check |
| --- | --- | --- |
| Python 3.11+ (3.12 used here) | runtime | `python3 --version` |
| [`uv`](https://docs.astral.sh/uv/) | package manager, runs every command | `uv --version` |
| PostgreSQL 16 (server + client) | picks / validations / ledger; `pg_dump` for backups | `psql --version`, `pg_dump --version` |
| [Claude Code](https://claude.com/claude-code) CLI, **logged in** | the pick validator shells out to `claude -p` | `claude --version`, then `claude -p "say ok" --model opus` |
| `git` + GitHub SSH key | clone `git@github.com:Murnster/betting-agent.git` | `ssh -T git@github.com` |
| `cron` | the loop | `crontab -l` |

Accounts you already have — copy the values from this machine's `.env`:

- **The Odds API** key (free tier, 500 credits/month — see §7 for how the loop spends them).
- **Discord** webhook URLs for the NFL picks channel and the NFL results channel.
- Claude login is per-machine: run `claude` once interactively and sign in. Do **not** use `--bare` anywhere; it skips the keychain and reports "Not logged in".

Ollama, OpenWeatherMap, Gemini and Tavily are **not** needed for the NFL loop.

### Time zone

`props.py --today` picks the slate by the **machine's local calendar day** and the cron
schedule below assumes the machine runs in a North American zone (this box is
`America/Halifax`, ET+1). Either set the new laptop to the same zone:

```bash
sudo timedatectl set-timezone America/Halifax
```

or shift every hour in the crontab by your offset. In a far-east zone (e.g. Australia)
the Sunday slate is *Monday* locally and Thursday night is *Friday*; the day-of-week
fields must move too. Setting the zone is the simpler path.

---

## 2. Clone and install

```bash
git clone git@github.com:Murnster/betting-agent.git ~/source/betting-agent
cd ~/source/betting-agent
uv sync                      # creates .venv with every dependency, ~1 min
```

`uv sync` is all the Python setup there is. Never `pip install` into the venv by hand —
`uv add <package>` if something is missing.

---

## 3. PostgreSQL

Create the role and database the default `DATABASE_URL` expects
(`postgresql://postgres:postgres@localhost:5432/betting_agent`):

```bash
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres createdb betting_agent
```

Then apply the schema (migrations run through `f2a3b4c5d6e7`):

```bash
uv run alembic upgrade head
```

### Carry the pick history over (recommended)

The paper trade's whole point is the running record. On **this** machine:

```bash
uv run python scripts/backup_db.py            # writes backups/postgres/betting_agent_<stamp>.dump
```

Copy the newest `.dump` to the new laptop, then (instead of `alembic upgrade head` on an
empty DB, or after dropping and recreating it):

```bash
pg_restore --clean --if-exists --no-owner -d betting_agent backups/postgres/betting_agent_<stamp>.dump
uv run alembic upgrade head                   # no-op if the dump is current; safe either way
```

The dump is `pg_dump --format=custom`, so `pg_restore` is the tool, not `psql`.
Do not run the loop on both machines against separate databases — you would get two
diverging ledgers and two Discord posts per slate. Stop cron here once the new laptop is
live (`crontab -r` on this box removes only the loop if nothing else is scheduled).

---

## 4. `.env`

Start from the template and fill in the NFL block. These are the only keys the loop
reads; everything else in `.env.example` is for the frozen sports.

```bash
cp .env.example .env
```

```ini
# --- required ---
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/betting_agent
ODDS_API_KEY=<your key>

# --- Discord (the only way picks are seen on an unattended box) ---
DISCORD_ENABLED=true
DISCORD_WEBHOOK_NFL_PICKS=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_NFL_RESULTS=https://discord.com/api/webhooks/...

# --- books: The Odds API has NO bet365 player props; DraftKings/FanDuel price the card ---
# PREFERRED_BOOKMAKERS=                      (default empty; the props path
#                                             falls back to bet365 on its own)
# PROP_FALLBACK_BOOKMAKERS=draftkings,fanduel (default)

# --- paper bankrolls ---
STARTING_BANKROLL=1000.0          # main props card
LADDER_BANKROLL=100.0             # ladder hits (own book)
OVERS_BANKROLL=100.0              # straight overs (own book)

# --- sections (all on by default) ---
# TD_PROPS_ENABLED=true           # anytime-TD scorer, +1 credit/game
# LADDER_ENABLED=true             # alternate boards, +3 credits/game
# LADDER_MARKETS=player_receptions,player_reception_yds,player_rush_yds
# OVERS_ENABLED=true              # main-line overs, free

# --- validator: Opus via the local claude CLI, every game, nothing skipped ---
AGENT_ENABLED=true
AGENT_MODEL=claude/opus
AGENT_MODE=all                    # 'top' would leave games past the cap SKIPPED
AGENT_SHADOW=true                 # verdicts recorded and shown, stakes untouched
AGENT_CLAUDE_WEB_SEARCH=true
AGENT_DAILY_BUDGET_USD=100.00
AGENT_CLAUDE_MAX_CALL_USD=10.00
AGENT_CLAUDE_TIMEOUT=900
AGENT_CLAUDE_MAX_TURNS=16
AGENT_RETRIES=1
```

The validator caps are deliberately wide (owner's call: a skipped validation must never
come into play, and time does not matter). Read the `Validator: … $x.xxxx` line in
`logs/nfl_card.log` after the first Sunday before deciding whether to tighten them.
Keep `AGENT_SHADOW=true` until the `agent_validations` table shows the verdicts add
value against graded results.

---

## 5. Verify before scheduling

Run these in order. Steps 1–3 are free.

```bash
uv run pytest tests/ -q                       # ~600 tests, ~12 s
uv run python scripts/props.py --today        # off-day → "No NFL games today", 0 credits
uv run python scripts/props.py --closing      # nothing due → exits, 0 credits
uv run python scripts/grade.py --sport NFL    # grades whatever is pending, prints the ROI report
```

Then one paid dry run against the next game on the schedule (6 credits + 3 for the lean,
one Opus call), which also proves the Discord webhooks:

```bash
uv run python scripts/props.py --save --max-events 1
```

You should see the card in the terminal and in the picks channel: game lean, TD scorer,
props, `STRAIGHT OVERS`, `LADDER HITS`, each entry carrying a validator verdict tagged
`(SHADOW)`. If a verdict says `SKIPPED`, check `claude -p` works from a plain shell
(`claude -p "say ok" --model opus`) — that is the only reason it should.

`--save` refreshes prices on already-saved ungraded picks and never duplicates them, so
running the dry run on a machine restored from this box's dump is safe.

---

## 6. Schedule it

`scripts/nfl_loop.sh` is the single cron entry point. It finds `uv` and `claude` in
`~/.local/bin` (override with `UV_BIN` / `CLAUDE_BIN_DIR`), logs to `logs/`, and prints its
own crontab:

```bash
scripts/nfl_loop.sh crontab       # review the lines
scripts/nfl_loop.sh crontab | crontab -    # install (REPLACES the user's crontab)
crontab -l
```

If the user already has a crontab, append with `crontab -e` instead of piping.

Two knobs, set in the crontab environment or before the command:

- `NFL_SUGGEST` (default 16): games priced on a full slate, ranked by model heat. 16 is
  the largest slate of the season, so every game of every day is priced; see §7.
- `NFL_MAX_PICKS` (default 10): main-card prop candidates kept **per slate**.
- `NFL_CLOSING_WINDOW` (default 90 minutes).

**`NFL_SUGGEST` is what keeps every Sunday window on the card, and lowering it can
silently drop a whole window.** `--suggest` heat-ranks the day in one flat list that has
no idea slates exist; the split into Sunday Early / Sunday Late / Sunday Night happens
afterwards. Spend all six games on the 1pm window and the 4pm window and Sunday night
have nothing to card — and `props.py` does not post a slate with nothing on it. Sunday
night is the most exposed: one game competing against a full slate.

`--max-picks` used to have the same failure mode (a single global cut by edge) and is now
applied per slate (`cap_per_slate`), so each window keeps its own allowance. Raising it
does **not** add card slots — those are the per-slate caps in `intelligence/slate.py`
(3 props in a window, 2 in a primetime single-game slate) — it only adds off-card saved
picks. Off-card picks are staked in the paper ledger like any other, so this is a real
change in exposure, not just bookkeeping.

The laptop has to be awake at those times. Disable suspend on AC power (GNOME: Settings
→ Power → Automatic Suspend off; or `sudo systemctl mask sleep.target suspend.target`)
and keep it plugged in. Cron does not run missed jobs — a `card` run that is missed
means no picks for that slate.

---

## 7. Odds API credit budget (500/month per key)

Per game the card costs **6 credits**: receptions + receiving yards (2), anytime TD (1),
three alternate boards (3). The game lean is 3 credits per run regardless of games.
`--closing` re-fetches the markets for each game with held picks (6 per game, once —
`capture_closing_lines_for_upcoming` filters on `Pick.closing_odds IS NULL`, so the
hourly job never re-bills a game it already captured).

**One free key does not cover the season.** Costed against the real 2026 schedule with
every section on and every game priced (`NFL_SUGGEST=16`), cards plus the daily lean call
plus one closing capture per game:

| Month | Games | Game days | Credits |
| --- | --- | --- | --- |
| Sep 2026 | 48 | 10 | 414 |
| Oct 2026 | 60 | 13 | 519 |
| Nov 2026 | 71 | 16 | **616** |
| Dec 2026 | 62 | 15 | 541 |
| Jan 2027 | 31 | 3 | 257 |

Every month except January is at or over a single key's 500. When a key runs out the API answers
401 and the card job dies with `No prop odds returned` — no Discord post, and nothing
tells you except the absence of a card.

**The fix is more keys, not a thinner card.** Set `ODDS_API_KEY_2` and `ODDS_API_KEY_3`
to additional Odds API accounts; `OddsAPIClient` walks them in order, rotating on a
quota 401/429 (and on a revoked key), and retires a key the moment its
`x-requests-remaining` header hits zero so the next call does not waste a round trip.
Three free keys is a 1500/month pool against a 616-credit peak — 2.4x headroom while
still pricing every game of every slate. Asking an exhausted key
costs nothing, so the fallback itself is free. Each cron run is a fresh process and
starts again at key 1: expect one harmless 401 per run late in a month.

Every fetch logs `Odds API key N of M: X credits remaining` at INFO, so
`grep 'credits remaining' logs/nfl_card.log | tail` is the monthly health check.

Pricing more games costs credits but barely touches the Opus bill: the validator only
ever sees card entries (`props.py` passes `on_cards`), and those are capped per slate, so
a wider fetch buys a better field to choose the same number of card picks from.

If you would rather run on one key, the levers in order of what you give up least are —
but note the first one costs you Sunday windows, see §6: lower `NFL_SUGGEST`;
`LADDER_MARKETS` without
`player_receptions` (-1/game, and the diagnostic rates receptions ladders worst);
`--no-td` (-1/game); `--no-leans` (-3/run).

---

## 8. Day-to-day

- **Picks**: the Discord card. The book shown is the one the price came from (DraftKings
  or FanDuel); check bet365's number by hand before staking anything real.
- **Record a real bet**: `uv run python scripts/bets.py set <pick_id> --stake 25 --odds -115`
  (`bets.py list` shows ids). Paper picks otherwise stay paper.
- **Results**: posted to the results channel by the 09:00 grade run, with the three books
  on their own lines (props, straight overs, ladder hits) plus TD scorers and leans, and
  an all-time recap with each book's bankroll.
- **Reports**: `uv run python scripts/report.py --sport NFL` (ROI, CLV, bankroll);
  `uv run python scripts/bets.py ledger`.
- **Logs**: `logs/nfl_card.log`, `nfl_closing.log`, `nfl_grade.log`, `nfl_backup.log`.

### Things that look broken but are not

- `WARNING No player stats for 2026 … 404` until nflverse publishes week 1 — projections
  rest on 2024–25 until then.
- `Injury report unavailable … Season must be between 2009 and 2025` before the Thursday
  after Labor Day — nflreadpy's season gate; the Opus validator covers injuries meanwhile.
- `bet365 posted no props … priced against draftkings` on every game — expected; The Odds
  API has no bet365 prop feed.
- A Sunday's props stay ungraded on Monday morning — nflverse publishes the weekly stats
  parquet a day or two later; the 09:00 grade run picks them up when it appears.

### Updating the code

```bash
git pull && uv sync && uv run alembic upgrade head && uv run pytest tests/ -q
```

---

## 9. Reference

- `CLAUDE.md` — how every piece works (props model, TD board, ladder, overs, validator,
  gotchas). Read the "In-season NFL props loop" section first.
- `TODO.md` — open follow-ups and the review dates for the TD window, ladder and overs
  sections (4–6 weeks into the season).
- `scripts/props.py --help` — every flag (`--suggest`, `--pick-games`, `--no-td`,
  `--no-ladder`, `--no-overs`, `--no-leans`, `--agent-mode`, `--closing`).
