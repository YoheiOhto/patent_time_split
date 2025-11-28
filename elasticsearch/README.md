
# Elasticsearch README
This directory contains configuration and setup instructions for running Elasticsearch in a Docker container.
## Prerequisites
- Docker installed on your machine.
## Setup Instructions
1. Navigate to the `elasticsearch` directory:
2. make mount directory for elasticsearch data:
   ```shell
  sudo mkdir -p ~/esdata
  sudo chown -R 1000:1000 ~/esdata
  sudo chmod -R 755 ~/esdata
   ```
3. network settings:
```shell
docker network ls | grep stack
```
```shell
docker network create stack
```
4. use docker compose to start the Elasticsearch service:
   ```shell
   docker compose up -d
   ```
5. Verify that the Elasticsearch container is running:
```shell
curl -u elastic:micgm1Gemini http://localhost:9200
{
  "name" : "5afc8253f71e",
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "nLdXTEOFTNqQAT7pNTqp8Q",
  "version" : {
    "number" : "8.13.2",
    "build_flavor" : "default",
    "build_type" : "docker",
    "build_hash" : "16cc90cd2d08a3147ce02b07e50894bc060a4cbf",
    "build_date" : "2024-04-05T14:45:26.420424304Z",
    "build_snapshot" : false,
    "lucene_version" : "9.10.0",
    "minimum_wire_compatibility_version" : "7.17.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "You Know, for Search"
}
```

# pubmed dataset 
this repository does not contain pubmed dataset due to its large size. please download the dataset from the following link:
https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
and extract the files with using the following repository:
https://github.com/YoheiOhto/pubmed_handler