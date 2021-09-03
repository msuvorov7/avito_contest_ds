#!/usr/bin/env bash

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1LpjC4pNCUH51U_QuEA-I1oY6dYjfb7AL' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LpjC4pNCUH51U_QuEA-I1oY6dYjfb7AL" \
-O train.tar.gz && rm -rf /tmp/cookies.txt && \
tar -zxvf train.tar.gz && rm -rf train.tar.gz
