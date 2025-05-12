#!/bin/bash

STUDENT_NAME="Zohaib Shaikh"
STUDENT_ID=6891120
BS=32
LR=0.0001

LOG_RSGNET="logs/${STUDENT_NAME// /_}_ID${STUDENT_ID}_rsgnet_train_${BS}_${LR}.json"
LOG_REMOVED="logs/${STUDENT_NAME// /_}_ID${STUDENT_ID}_rsgnet_removed_train_${BS}_${LR}.json"

get_qwk() {
  python3 -c "
import json
with open('$1') as f:
    print(json.load(f)['epoch_logs'][-1]['val_qwk'])
"
}

QWK_RSGNET=$(get_qwk "$LOG_RSGNET")
QWK_REMOVED=$(get_qwk "$LOG_REMOVED")

echo "🔍 Comparing QWK:"
echo "• rsgnet:         $QWK_RSGNET"
echo "• rsgnet_removed: $QWK_REMOVED"

if (( $(echo "$QWK_RSGNET > $QWK_REMOVED" | bc -l) )); then
  BEST_MODEL="rsgnet"
else
  BEST_MODEL="rsgnet_removed"
fi

echo "🏆 Best model: $BEST_MODEL → running avgpool test"

bash avg_pool_rsgnet_test.sh "${BEST_MODEL}_avg_best" $BS $LR none
