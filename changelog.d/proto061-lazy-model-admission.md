Cold model requests now enter the staging queue before their first construction loads.
Queue admission validates the selected binding and environment; device entry still
requires the loaded model. Known construction failures retain their original cause
instead of reporting a misleading unknown binding.
