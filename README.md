# RSE3241
Hydropower

## Streamlit Community Cloud

- Entrypoint: `hydro-power.py`
- Deploy with Python 3.12 selected in **Advanced settings**.
- Python packages are pinned in `requirements.txt`; Linux TeX packages used by
  the optional PDF export are listed in `packages.txt`.

An existing Community Cloud app cannot change Python version in place. If its
logs show another version, delete and redeploy the app with Python 3.12, keeping
the same repository, branch, entrypoint, subdomain and secrets.
