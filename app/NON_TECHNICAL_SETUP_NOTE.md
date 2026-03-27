# Streamlit App Setup Note (Non-Technical)

## What happened
When we first launched the app, it failed at startup because one part of the model software (XGBoost) needs an extra system component on Mac called OpenMP (`libomp`).

On this computer, we cannot install that component because it is a managed/shared machine and does not allow the required system-level permissions.

## Why this is not your fault
Nothing is wrong with your data or notebook logic.

The issue is environmental:
- The model library expects a system file to exist.
- The machine blocks the install of that file.
- So the app cannot load XGBoost directly.

## Workaround we implemented
The app now has a built-in fallback model.

If XGBoost is available:
- It uses the tuned XGBoost model.

If XGBoost is not available:
- It automatically switches to a backup model (Gradient Boosting from scikit-learn).
- The app still runs, predicts prices, and shows diagnostics.
- The page also tells you when fallback mode is active.

## What this means for demos and presentations
You can still run and present the application without admin access.

The experience is still complete:
- Intro page
- Feature-adjustable prediction page
- Residual/diagnostics page
- Model summary page

## Accuracy expectations
The fallback model is strong, but it may be slightly less accurate than the tuned XGBoost model in some cases.

This is a practical tradeoff to keep the app usable on restricted machines.

## Simple explanation to share with others
"The original model needs a system dependency we cannot install on this managed computer. To keep the app working, we added an automatic backup model that runs without admin privileges. The app is fully functional and transparently shows which model is active."

## If running on a personal machine later
On a personal machine with admin rights, you can install the missing dependency and use XGBoost directly.

Typical steps:
1. Install OpenMP (`libomp`) on macOS.
2. Reinstall XGBoost in the virtual environment.
3. Re-run Streamlit.

The app will automatically use XGBoost when available.
