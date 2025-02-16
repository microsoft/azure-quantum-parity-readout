#!/bin/bash

LANGUAGE="$1"
NOTEBOOK="$2"

SPEC="{\"kernelspec\": {\"display_name\": \"$LANGUAGE\", \"language\": \"$LANGUAGE\", \"name\": \"$LANGUAGE\"}}"

if grep -q "^  jupytext:$" "$NOTEBOOK"; then
    jupytext $NOTEBOOK --update-metadata "$SPEC"
fi
