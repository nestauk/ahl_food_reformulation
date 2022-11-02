# Identifying targets for reformulation using the Kantar dataset

**_Repository for hosting the analysis that contributes to the report [insert link]_**

## Welcome :wave:

Intro text

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `git-crypt`, `direnv`, and `conda`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS

## Installation and running the code

### Clone and set up the repo

1. Follow the below steps to setup the project.

```shell
$ git clone https://github.com/nestauk/ahl_food_reformulation
$ cd ahl_food_reformulation
```

2. Run the command `make install` to create the virtual environment and install dependencies

### Download and save the data

Download the data from the `latest_data` folder in [ahl-private-data](https://s3.console.aws.amazon.com/s3/buckets/ahl-private-data?prefix=kantar%2Fdata_v3%2F&region=eu-west-2&showversions=false#) s3 bucket and save to following folders:

- `inputs/data`
- `outputs/data`

The data in `latest_data` should be inside `inputs` and `outputs` folders.
