import os
import requests
from icoscp_core.icos import bootstrap
import zipfile

cookie_token = "WzE3NTMzODQ3OTc0NDQsInNhbWFudGhhLmJpZWdlbEBpbmYuZXRoei5jaCIsIlNhbWwiXR4wZQIwIacY2dykV3z5IdtfAC1oeI+dbq5GRf4wVxk1zPvFL/D2z86NoK+opnLV+GqcOWKeAjEA7428CC9Klwsp0aNEaJ6NnXW+O+/Wwx5IoIThYEhSLBT7VVDj9nlwFxCfNfpO082O"
meta, data = bootstrap.fromCookieToken(f"cpauthToken={cookie_token}")

collection_uri = "https://meta.icos-cp.eu/collections/mdtEHjyujUDsC9vgMv5eeH8B"

coll_json = requests.get(collection_uri, headers={"Accept":"application/json"}).json()

members = coll_json["members"]

out_dir = "/cluster/work/igp_psr/sbiege/gpp-ml/fluxneteo"
os.makedirs(out_dir, exist_ok=True)

for m in members:
    handle = m["hash"]
    obj_meta = requests.get(m["res"], headers={"Accept":"application/json"}).json()
    fn = obj_meta['fileName']
    url = obj_meta.get("accessUrl")
    local_path = os.path.join(out_dir, fn)
    if os.path.exists(local_path):
        continue

    with requests.get(url,
                        stream=True,
                        cookies={"cpauthToken": cookie_token}) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    with zipfile.ZipFile(local_path, 'r') as z:
        z.extractall(out_dir)
    os.remove(local_path)

    print(f"Downloaded {fn}")