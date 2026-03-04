from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from scraper import scrape_website
from analyzer import analyze

load_dotenv()

app = FastAPI(title="AI Readiness Scanner")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/scan", response_class=HTMLResponse)
async def scan(request: Request, url: str = Form(...), provider: str = Form(...)):
    try:
        content = await scrape_website(url)
        report = await analyze(content, provider)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "report": report,
            "scanned_url": url,
        })
    except ValueError as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Analysis failed: {str(e)}",
        })
