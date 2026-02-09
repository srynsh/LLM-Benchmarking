#!/bin/bash
# filepath: /Users/umair/projects/codaveri/llm-benchmarking/generate_data_manifest.sh

find data -type f -exec shasum -a 256 {} \; | sort > data_manifest.sha256