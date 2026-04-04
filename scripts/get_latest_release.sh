#!/bin/sh
# CREATED:2023-06-02 12:31:15 by Brian McFee <brian.mcfee@nyu.edu>
#   A shell script to quickly pull the most recent proper release tag
#   This is used by the multi-version documentation build to automatically
#   infer which doc site to consider "latest"

latest_release=$(git tag -l --sort=creatordate | egrep -v "rc|pre|post" | tail -1)

if [ -n "$latest_release" ]; then
	printf '%s\n' "$latest_release"
else
	printf 'main\n'
fi
