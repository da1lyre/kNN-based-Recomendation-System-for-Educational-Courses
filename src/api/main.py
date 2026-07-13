from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from src.core.model import load_model

recommender, df = load_model()

app = FastAPI(
    title="Course Recommender System",
    description="k-NN based course recommendations",
    version="1.0.0"
)

class RecommendRequest(BaseModel):
    course_titles: List[str]

with open("../../app/html/home.html", "r", encoding="utf-8") as page:
    HTML_PAGE = page.read()

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML_PAGE)


@app.post("/recommend")
def get_recommendations(request: RecommendRequest):
    try:
        recs = recommender.recommend_by_titles(request.course_titles)
        return {
            "recommendations": recs[
                ["course_title", "course_organization", "course_difficulty", "course_rating", "source"]].to_dict(
                "records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))