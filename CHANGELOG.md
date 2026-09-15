# Changelog

User-facing notes, written to be posted straight into the general chat. One
section per day, newest first. Bullets say what changed on the card, in the
results post or in the reports — one or two lines each, no rationale; the
commit messages carry that. Group under **New**, **Changed** and **Fixed**,
drop any empty heading, and skip anything a reader of the Discord channels
would never notice (refactors, test data, lint).

Keep each bullet on ONE line, however long — Discord renders hard wraps oddly
— and stick to plain hyphens over unicode minus signs.

To post the newest section to the general chat (`DISCORD_WEBHOOK_CHANGELOG`
in `.env`; Discord caps a message at 2000 characters):

    awk '/^## /{n++} n==1' CHANGELOG.md | jq -Rs '{content: .}' | \
      curl -sS -X POST -H 'Content-Type: application/json' -d @- \
      "$(grep '^DISCORD_WEBHOOK_CHANGELOG=' .env | cut -d= -f2-)"

## 2026-09-15

**Changed**
- Parlay legs are now picked by how likely they are, not by claimed edge: everything the model has at 60%+ goes first, and only if a game can't fill three legs does it drop to the next most likely.
- A ladder rung needs 50%+ from the model to be a parlay leg, same rule the TD scorers already had - last night's ticket carried a 46% rung ahead of a 59% receiver.
- Parlay tickets now aim for +200 rather than +300, so reaching for the price pulls in fewer long shots.
- Ladder picks need a bigger edge to make the card: 8% on rushing yards, 10% on receiving yards, up from 3%. Fewer rungs, and the walk-forward has both the hit rate and the ROI going up.
- The receptions ladder is off - it lost money at every edge floor we tested, and tight ends were the worst of it (24.7% hits against 49.6% claimed). Saves a credit a game.
- A game no longer gets a ladder pick just for being on the slate. The best rung used to be kept whatever its edge, which is where the worst four picks came from.

## 2026-09-14

**New**
- Long-shot parlays: $1 tickets in their own channel and their own paper book — same-game per primetime game, cross-game per Sunday window, plus a 3-5 leg ML/spread/total lean. Costs no credits, never touches the main record.
- Same-game prices shown are the product of the legs — a real book pays less, so check bet365 by hand.
- Tickets take at most one pick off the tracked card, prefer props to game lines, and only take a TD scorer we have at 50%+.

**Changed**
- TD scorers are their own paper book: below the props, blue, own $100 bankroll and own record line. Green is the main card only.
- The validator's "Why" now argues the pick — production, role, matchup — instead of reading like an injury report, and isn't cut off mid-sentence.
- Morning grading moved from 09:00 to 08:00.

**Fixed**
- The weekly report headline was blending all four books into one record (18-34 / -16.7%) when the main book was 8-7 / +11.1%. Now scoped, with each side book listed separately.
- The results post's breakdown embed had the same problem, and now matches the headline above it.
- A second ladder rung on the same player could show a stale opponent.

Grading is unchanged throughout — only the record and bankroll picks count under moved.
