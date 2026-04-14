"""
Analysis Routes — Retrieve analyzed call data and dashboard stats.
"""
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.get("/calls")
async def list_calls():
    """List all analyzed calls (summary view)."""
    from backend.routes.upload import analyzed_calls
    
    calls_list = []
    for call_id, call in analyzed_calls.items():
        calls_list.append({
            "id": call["id"],
            "filename": call["filename"],
            "timestamp": call["timestamp"],
            "duration": call.get("duration", 0),
            "call_score": call["call_score"],
            "overall_sentiment": call["overall_sentiment"],
            "conversion_probability": call["conversion_probability"],
            "objection_count": len(call["all_objections"]),
            "magic_moment_count": len(call["magic_moments"])
        })
    
    # Sort by timestamp descending
    calls_list.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"calls": calls_list, "total": len(calls_list)}


@router.get("/calls/{call_id}")
async def get_call(call_id: str):
    """Get full analysis for a specific call."""
    from backend.routes.upload import analyzed_calls
    
    if call_id not in analyzed_calls:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")
    
    return analyzed_calls[call_id]


@router.get("/stats")
async def get_dashboard_stats():
    """Aggregate statistics for the dashboard."""
    from backend.routes.upload import analyzed_calls
    
    if not analyzed_calls:
        return {
            "total_calls": 0,
            "avg_score": 0,
            "avg_sentiment": 0,
            "avg_conversion_probability": 0,
            "top_objections": [],
            "score_distribution": {"excellent": 0, "good": 0, "average": 0, "poor": 0},
            "recent_calls": []
        }
    
    calls = list(analyzed_calls.values())
    
    # Averages
    avg_score = sum(c["call_score"] for c in calls) / len(calls)
    avg_sentiment = sum(c["overall_sentiment"]["score"] for c in calls) / len(calls)
    avg_conversion = sum(c["conversion_probability"] for c in calls) / len(calls)
    
    # Objection frequency
    objection_counts = {}
    for call in calls:
        for obj in call["all_objections"]:
            cat = obj["category"]
            objection_counts[cat] = objection_counts.get(cat, 0) + 1
    
    top_objections = sorted(
        [{"category": k, "count": v} for k, v in objection_counts.items()],
        key=lambda x: x["count"],
        reverse=True
    )
    
    # Score distribution
    score_dist = {"excellent": 0, "good": 0, "average": 0, "poor": 0}
    for call in calls:
        score = call["call_score"]
        if score >= 80:
            score_dist["excellent"] += 1
        elif score >= 60:
            score_dist["good"] += 1
        elif score >= 40:
            score_dist["average"] += 1
        else:
            score_dist["poor"] += 1
    
    # Recent calls
    recent = sorted(calls, key=lambda x: x["timestamp"], reverse=True)[:5]
    recent_calls = [{
        "id": c["id"],
        "filename": c["filename"],
        "timestamp": c["timestamp"],
        "call_score": c["call_score"],
        "overall_sentiment": c["overall_sentiment"]["label"]
    } for c in recent]
    
    return {
        "total_calls": len(calls),
        "avg_score": round(avg_score, 1),
        "avg_sentiment": round(avg_sentiment, 3),
        "avg_conversion_probability": round(avg_conversion, 2),
        "top_objections": top_objections,
        "score_distribution": score_dist,
        "recent_calls": recent_calls
    }
