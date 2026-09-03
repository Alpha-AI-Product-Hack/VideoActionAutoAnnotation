# Data Sample

## Input

### rules.json

```json
{
    "actions": [
        "pick_up", 
        "move", 
        "put_down"
    ],
    "objects": [
        "cup",
        "glass"
    ],
    "min_duration_ms": 500,
    "min_confidence": 0.7
}
```

## Output

### actions.json

```json
[
  {
    "id": "1",
    "start_ms": 1200,
    "end_ms": 3800,
    "action": "pick_up",
    "object": "cup",
    "keyframe_ms": 2500,
    "confidence": 0.94,
    "model_version": "pipeline-0.1"
  },
  {
    "id": "2",
    "start_ms": 4500,
    "end_ms": 7200,
    "action": "move",
    "object": "cup",
    "keyframe_ms": 5800,
    "confidence": 0.87,
    "model_version": "pipeline-0.1"
  },
  {
    "id": "3",
    "start_ms": 8100,
    "end_ms": 10400,
    "action": "put_down",
    "object": "cup",
    "keyframe_ms": 9200,
    "confidence": 0.91,
    "model_version": "pipeline-0.1"
  }
]
```