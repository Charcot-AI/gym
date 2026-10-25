# Submissions

Only publicly available model APIs or weights are ranked. Each submission includes a submitter, organization, ISO submission date, and availability (`api`, `weights`, or `unreleased`). Submit one result per model version. Organization counts are reported per leaderboard selection.

Private evaluation output contains only `summary.json`. For leaderboard intake, place a copy of that aggregate summary beside its `submission.json`; do not submit episode records or trajectories. Use the published runner version to reproduce each result.

Each leaderboard input directory needs a `submission.json` file beside its summary:

```json
{
  "submitter": "Example Submitter",
  "organization": "Example Organization",
  "submitted": "2026-10-01",
  "availability": "api"
}
```

`unreleased` submissions remain visible without a rank. A leaderboard rejects duplicate model versions and missing or invalid submission metadata.
