# xt-vaep

End-to-end football analytics for possession value:
- **xT (Expected Threat)** from state-value iteration on pass/carry transitions and shot/goal likelihoods
- **VAEP-lite** via two classifiers for near-term score/concede risk

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip -r requirements.txt
pip install -e .
make all
make app
