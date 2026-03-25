# Dataset Information

This directory contains the evaluation datasets for MAGMA system.

## Datasets

### 1. LoCoMo Dataset
- **File**: `locomo10.json` (✅ included)
- **Description**: Long Conversation Memory dataset with 10 conversation samples
- **Format**: JSON with conversation turns and Q&A pairs
- **Categories**: Multi-hop, Temporal, Open-domain, Single-hop, Adversarial
- **Size**: 10 samples, ~1000 Q&A pairs, 2.7MB

### 2. LongMemEval Dataset
- **File**: `longmemeval_s_cleaned.json` (⬇️ download required)
- **Description**: Multi-session conversation memory evaluation dataset
- **Format**: JSON with sessions and counting/preference questions
- **Focus**: Multi-session counting and fact aggregation
- **Size**: ~265MB
- **Source**: HuggingFace - xiaowu0162/longmemeval-cleaned

#### Download Instructions:
```bash
# From the repository root directory
mkdir -p data/
cd data/
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
cd ..
```

**Note**: A sample dataset is provided in `../examples/longmemeval_sample.json` for quick testing

## Dataset Structure

### LoCoMo Format
```json
{
  "samples": [
    {
      "conversation": [
        {"speaker": "User", "text": "..."},
        {"speaker": "Assistant", "text": "..."}
      ],
      "questions": [
        {
          "question": "What did the user mention?",
          "answer": "Expected answer",
          "category": 1
        }
      ]
    }
  ]
}
```

### LongMemEval Format
```json
{
  "sessions": [
    {
      "session_id": "session_1",
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
  ],
  "questions": [
    {
      "question": "How many items were mentioned?",
      "answer": 5,
      "question_type": "multi_session_count",
      "session_id": "1-3"
    }
  ]
}
```

## Obtaining Datasets

Due to size and licensing, the full datasets are not included. You can:

1. Use the provided sample datasets in `examples/`
2. Contact the original dataset authors for full versions
3. Create your own test data following the formats above

## Sample Data

See the `examples/` directory for small sample datasets to test the system.