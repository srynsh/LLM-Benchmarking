#!/bin/bash

find data -type f ! -name '.DS_Store' -exec shasum -a 256 {} \; | awk '{print $2, $1}' | sort > data_manifest.sha256