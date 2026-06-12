# MDD Triangle + MoodGarden EMA Demo Deployment

This Streamlit app is prepared for public demo deployment.

## Public Mode

Public deployments use CSV upload for MoodGarden EMA data.

- Main app file: `mdd_streamlit_app.py`
- Requirements file: `requirements.txt`
- Required data file: `mdd_triangle_circuit_map.json`
- Optional helper module: `ema_integration.py`

The local MoodGarden live-file bridge is hidden by default in public mode.

## Local Live Sync

For local development only, enable the MoodGarden live JSON bridge with:

```bash
ENABLE_LOCAL_LIVE_SYNC=1 streamlit run mdd_streamlit_app.py
```

Public servers should not enable this flag because `moodgarden_live.json`
is a local-only bridge file.

## Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Create a new Streamlit app.
3. Set the main file path to `mdd_streamlit_app.py`.
4. Confirm `requirements.txt` is at the repository root.

## Privacy

Do not commit personal MoodGarden CSV exports or live JSON files.
Users should upload CSV files in their own browser session.
