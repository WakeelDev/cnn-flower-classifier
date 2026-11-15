import h5py

with h5py.File("mobilenetv2_model.h5", "r+") as f:
  config_attr = f.attrs["model_config"]

  # Handle both bytes and str attribute types across h5py versions
  if isinstance(config_attr, bytes):
    conf = config_attr.decode("utf-8")
  else:
    conf = str(config_attr)

  # Replace the list axis with integer axis format
  patched_conf = conf.replace('"axis": [3]', '"axis": 3')

  if isinstance(config_attr, bytes):
    f.attrs["model_config"] = patched_conf.encode("utf-8")
  else:
    f.attrs["model_config"] = patched_conf

print("Successfully patched mobilenetv2_model.h5!")