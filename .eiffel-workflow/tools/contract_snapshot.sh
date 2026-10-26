#!/usr/bin/env bash
# Dump every assertion clause (require / ensure / invariant blocks, full text) of src/**/*.e,
# one file section at a time, so a Phase 4 diff catches ANY changed clause, not just keywords.
# Usage: contract_snapshot.sh <src-dir>
src="$1"
find "$src" -name '*.e' | sort | while read -r f; do
	echo "=== ${f#$src/}"
	tr -d '\r' < "$f" | awk '
		/^[ \t]*(require|ensure|invariant)([ \t]|$)/ { inside = 1; print; next }
		inside && /^[ \t]*(do|once|deferred|external|local|alias|end|attribute)([ \t]|$)/ { inside = 0; next }
		inside && /^(feature|end)/ { inside = 0; next }
		inside { print }
	'
done
