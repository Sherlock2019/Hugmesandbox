#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
REPO_PATH="${REPO_PATH:-$HOME/credit-appraisal-agent-poc}"
REMOTE_URL_SSH="git@github.com:Sherlock2019/credit-appraisal-agent-poc.git"
BRANCH="${BRANCH:-main}"
DATE_TODAY=$(date +"%Y-%m-%d")
TMP_FILE="/tmp/git_today_changes.txt"

cd "$REPO_PATH"

echo "📦 Repo: $REPO_PATH"
echo "🌞 Detecting files modified today: $DATE_TODAY"

# ─────────────────────────────
# FIND MODIFIED FILES TODAY
# ─────────────────────────────
find . -type f -newermt "$DATE_TODAY 00:00:00" ! -newermt "$DATE_TODAY 23:59:59" \
    -not -path "*/.git/*" > "$TMP_FILE"

if [[ ! -s "$TMP_FILE" ]]; then
    echo "✅ No files modified today. Nothing to push."
    exit 0
fi

echo "📝 Files to push today:"
cat "$TMP_FILE"
echo "──────────────────────────────"

# ─────────────────────────────
# ADD + COMMIT
# ─────────────────────────────
while read -r file; do
    git add "$file"
done < "$TMP_FILE"

COMMIT_MSG="🚀 Auto-push: files modified on $DATE_TODAY"
echo "💬 Commit message: $COMMIT_MSG"

git commit -m "$COMMIT_MSG" || echo "⚠️ No changes staged (maybe already committed)."

# ─────────────────────────────
# PUSH VIA SSH
# ─────────────────────────────
echo "🔗 Setting remote to SSH: $REMOTE_URL_SSH"
git remote set-url origin "$REMOTE_URL_SSH"

echo "⬆️ Pushing to branch: $BRANCH ..."
git push origin "$BRANCH"

echo "✅ Push complete!"

