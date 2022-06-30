import os
import yaml
import tarfile
import requests

# CHANGE THESE VARIABLES TO AUTO GENERATE BLUEPRINT
base_url = "https://metacloud.staging-cloud.cnvrg.io/marketplace/api/v1"
#base_url = "https://app.asdfcvtyncfshna4zkblbpt.staging-cloud.cnvrg.io/"
api_token = "ZkFPc0h3QnhEQk03T1ptMDZEeUZGQT09OlBNY1pSdjFvMmkzVUNBa0F5akxZUmc9PQ=="
default_version = "1.0.1"
admin_token = 'QmMxOiNLeXJBS2xnIWNnVnpmbytUc0wr'
library_files = ['Endpoint']
blueprint_file = "blueprint.yaml"
blueprint_readme_file = 'readme.md'

#library_files = ['s3_connector',
#                 'OCR Train',
#                 'OCR Inference'
#                 ]
#blueprint_file = "train_blueprint.yaml"
#blueprint_readme_file = 'train_README.md'

headers = {"api-token": api_token}

# GENERATE LIBRARIES
def create_library_version(library_name, path):
    with open(f"{path}/library.yaml", 'r') as f:
        schema = f.read()
        schema_dict = yaml.safe_load(schema)
        schema_dict["version"] = default_version
        schema = yaml.dump(schema_dict)

    with open(f"{path}/README.md", 'r') as f:
        readme = f.read()

    with open(f"{path}/requirements.txt", 'r') as f:
        dependencies = f.read()

    payload = {
        "data": {
            "type": "string",
            "attributes": {
                "schema_file": {
                    "raw": schema
                },
                "description": {
                    "raw": readme,
                    "content_type": "text/plain"
                },
                "dependency": {
                    "raw": dependencies,
                    "language": "python"
                }
            }
        }
    }

    response = requests.request(
        method="POST",
        url=f"{base_url}/libraries/{library_name}/versions/",
        headers=headers,
        json=payload
    )

    if response.status_code == 201:
        print(f"Library version for {library_name} created")
    else:
        print(f"Failed creating version for {library_name}")
        print(response.text)


def create_library(library_name):
    payload = {
        "data": {
            "attributes": {
                "name": library_name,
                "slug": library_name,
                "public": True,
            }
        }
    }

    response = requests.request("POST", f"{base_url}/libraries/", json=payload, headers=headers)

    if response.status_code == 201:
        print(f"Library {library_name} created")
    else:
        print(f"Failed creating {library_name} library")
        print(response.text)

def upload_library_version(library_name, path):
    library_file = "library.tar.gz"

    with tarfile.open(library_file, "w:gz") as tar:
        tar.add(path, arcname=os.path.basename(path))

    files = {'file': open(library_file, 'rb')}

    response = requests.request(
        method="PUT",
        url=f"{base_url}/libraries/{library_name}/versions/{default_version}",
        headers=headers,
        files=files
    )

    if response.status_code == 204:
        print(f"Library files for {library_name} were uploaded")
    else:
        print(f"Failed upload for {library_name}")
        print(response.text)

# CREATE BLUEPRINT
def create_blueprint_version(blueprint_name):
    with open(blueprint_file, 'r') as f:
        schema = f.read()
        schema_dict = yaml.safe_load(schema)
        schema_dict["version"] = default_version

        for library in schema_dict["tasks"]:
            library["library_version"] = default_version

        schema = yaml.dump(schema_dict)

    with open(blueprint_readme_file, 'r') as f:
        readme = f.read()

    # create version
    payload = {
        "data": {
            "attributes": {
                "schema_file": {
                    "raw": schema
                },
                "description": {
                    "raw": readme,
                    "content_type": "text/plain"
                }
            }
        }
    }

    response = requests.request(
        method="POST",
        url=f"{base_url}/blueprints/{blueprint_name}/versions/",
        headers=headers,
        json=payload
    )

    if response.status_code == 201:
        print(f"Blueprint version for {blueprint_name} created")
    else:
        print(f"Failed creating bluerint version for {blueprint_name}")
        print(response.text)

def create_blueprint(blueprint_name):
    payload = {
        "data": {
            "attributes": {
                "name": blueprint_name,
                "public": True
            }
        }
    }

    response = requests.request("POST", f"{base_url}/blueprints/", json=payload, headers=headers)

    if response.status_code == 201:
        print(f"Blueprint {blueprint_name} created")
    else:
        print(f"Failed creating {blueprint_name} blueprint")
        print(response.text)

def build_blueprint(blueprint_name):
    admin_headers = {"api-token": admin_token}
    response = requests.request("POST", f"{base_url}/blueprints/{blueprint_name}/build", headers=admin_headers)

    if response.status_code == 202:
        print(f"Blueprint {blueprint_name} is being built")
    else:
        print(f"Failed building {blueprint_name} blueprint")
        print(response.text)


for library_path in library_files:
    with open(f"{library_path}/library.yaml", 'r') as f:
        schema_yaml = f.read()
        schema_json = yaml.safe_load(schema_yaml)

    library_name = schema_json["title"].replace(" ", "-").lower()
    create_library(library_name)
    create_library_version(library_name, library_path)
    upload_library_version(library_name, library_path)


with open(blueprint_file, 'r') as f:
    schema_yaml = f.read()
    schema_json = yaml.safe_load(schema_yaml)

blueprint_name = schema_json["title"]
blueprint_slug = blueprint_name.replace(" ", "-").lower()
create_blueprint(blueprint_name)
create_blueprint_version(blueprint_slug)
build_blueprint(blueprint_slug)