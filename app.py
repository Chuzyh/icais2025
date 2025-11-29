import os
import json
import base64
from typing import Optional, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import httpx
import PyPDF2
from io import BytesIO
import numpy as np
import aiohttp
# Load environment variables
load_dotenv()

app = FastAPI(title="Science Arena Challenge Example Submission")

# Initialize AsyncOpenAI client for LLM models
client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

# Initialize AsyncOpenAI client for embedding model
embedding_client = AsyncOpenAI(
    base_url=os.getenv("SCI_EMBEDDING_BASE_URL"),
    api_key=os.getenv("SCI_EMBEDDING_API_KEY")
)


def extract_pdf_text_from_base64(pdf_b64: str) -> str:
    """
    Extract text from base64-encoded PDF using PyPDF2
    """
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return "\n".join(pages)

    except Exception as e:
        print(f"PDF parsing error: {str(e)}")
        return ""


async def get_embedding(text: str) -> List[float]:
    """
    Get embedding vector for text using embedding model
    """
    try:
        response = await embedding_client.embeddings.create(
            model=os.getenv("SCI_EMBEDDING_MODEL"),
            input=text
        )
        embedding = response.data[0].embedding
        # Log embedding results (truncated)
        print(f"[get_embedding] Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"[get_embedding] Embedding dimension: {len(embedding)}")
        print(f"[get_embedding] Embedding (first 5 values): {embedding[:5]}")
        return embedding
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    """
    if not vec1 or not vec2:
        return 0.0

    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for non-streaming endpoints
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )

async def get_keywords_from_model(query: str):
    prompt = f"""
    Task: Prepare keywords for a literature review search.

    Question: "{query}"

    Instructions:
    - Provide 2-3 relevant keywords for searching papers on api.openalex.org.
    - Output ONLY as a comma-separated list, like this:
      keyword1, keyword2, keyword3
    - Do NOT include any explanations, extra text, or JSON formatting.
    """

    print(f"[get_keywords_from_model] Query: {query}")

    stream = await client.chat.completions.create(
        model=os.getenv("SCI_LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
        stream=True
    )
    full_text = ""

    async for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                full_text += delta.content
    # async for chunk in stream:
    #     if chunk.choices and len(chunk.choices) > 0:
    #         delta = chunk.choices[0].delta
    #         if delta and getattr(delta, "reasoning_content", None):
    #             full_text += delta.reasoning_content       



    if "Keywords:" in full_text:
        content = full_text.split("Keywords:")[-1].strip()
    else:
        content = full_text.strip()

    keywords = [kw.strip() for kw in content.split(",") if kw.strip()]

    keywords = keywords[:6]

    print(f"🔍 Extracted Keywords: {keywords}")
    return keywords
async def get_reference_papers(query: str):
    prompt = f"""
    Task: Suggest reference papers for a literature review.

    Question: "{query}"

    Instructions:
    - Provide 4-5 relevant papers for this topic.
    - Output ONLY as a comma-separated list of paper titles.
    - DO NOT include author names, year, explanations, or extra text.
    - You can think step by step, but at the end output the final answer in ONE line, like:
      Title 1,  Title 2,  Title 3,  Title 4
    """
    print(f"[Prompt sent to model]\n{prompt}\n{'-'*50}")

    stream = await client.chat.completions.create(
        model=os.getenv("SCI_LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
        stream=True
    )

    full_text = ""
    print("[Model thinking...]")
    async for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                text = delta.content
                full_text += text
                # 实时打印中间过程
                print(text, end="", flush=True)

    print("\n" + "-"*50)
    print("[Full LLM response captured]")

    final_list = full_text.strip()
    if "Paper Titles:" in final_list:
        final_list = final_list.split("Paper Titles:")[-1].strip()
    titles = [t.strip() for t in final_list.split(",") if t.strip()][:5]
    print(f"📄 Suggested Papers: {titles}")
    return titles
OPENALEX_BASE = "https://api.openalex.org/works"

async def fetch_top_papers_for_keyword(keyword: str, per_type: int = 2):
    results = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1️⃣ 按引用数最多
        params_cited = {
            "search": keyword,
            "sort": "cited_by_count:desc",
            "per-page": per_type
        }
        resp_cited = await client.get(OPENALEX_BASE, params=params_cited)
        data_cited = resp_cited.json()
        results["most_cited"] = []
        for work in data_cited.get("results", []):
            authors = work.get("authorships", [])
            first_author = authors[0].get("author", {}).get("display_name") if authors else "未知作者"
            results["most_cited"].append({
                "title": work.get("title"),
                "doi": work.get("doi"),
                "publication_year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count"),
                "abstract": work.get("abstract_inverted_index"),
                "first_author": first_author
            })

        # 2️⃣ 按综合排序（Relevance）
        params_relevance = {
            "search": keyword,
            "sort": "relevance_score:desc",
            "per-page": per_type
        }
        resp_relevance = await client.get(OPENALEX_BASE, params=params_relevance)
        data_relevance = resp_relevance.json()
        results["most_relevant"] = []
        for work in data_relevance.get("results", []):
            authors = work.get("authorships", [])
            first_author = authors[0].get("author", {}).get("display_name") if authors else "未知作者"
            results["most_relevant"].append({
                "title": work.get("title"),
                "doi": work.get("doi"),
                "publication_year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count"),
                "abstract": work.get("abstract_inverted_index"),
                "first_author": first_author
            })

    return results


async def fetch_all_papers(keywords):
    tasks = [fetch_top_papers_for_keyword(kw) for kw in keywords]
    all_results = await asyncio.gather(*tasks)
    return {kw: res for kw, res in zip(keywords, all_results)}


async def summarize_single_paper_no_llm(paper):
    """
    不调用 LLM，直接返回摘要信息作为 summary。
    """
    title = paper.get("title", "N/A")
    year = paper.get("publication_year", "N/A")
    authors = ", ".join([a.get("author_name", "") for a in paper.get("authorships", [])]) \
             if paper.get("authorships") else "N/A"
    citations = paper.get("cited_by_count", 0)
    abstract = paper.get("abstract", "") or "No abstract available."

    # 直接用文章信息当 summary，确保兼容原结构
    summary = (
        f"Title: {title}\n"
        f"Year: {year}\n"
        f"Authors: {authors}\n"
        f"Citations: {citations}\n"
        f"Abstract Summary: {abstract}"
    )
    return summary
async def summarize_single_paper_with_llm(paper):
    client = AsyncOpenAI(
        base_url=os.getenv("SCI_MODEL_BASE_URL"),
        api_key=os.getenv("SCI_MODEL_API_KEY")
    )

    authors = ", ".join([a.get("author_name", "") for a in paper.get("authorships", [])]) if paper.get("authorships") else "N/A"
    abstract = paper.get("abstract", "")
    title = paper.get("title", "")
    year = paper.get("publication_year", "N/A")
    citations = paper.get("cited_by_count", 0)

    prompt = f"""
    Paper:
    Title: {title}
    Year: {year}
    Authors: {authors}
    Citations: {citations}
    Abstract: {abstract}

    Task: Write a concise, 3-4 sentence summary highlighting the key contributions or findings of this paper. 
    Only use information provided above.
    """

    response = await client.chat.completions.create(
        model=os.getenv("SCI_LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2
    )

    summary = response.choices[0].message.content
    return summary
async def summarize_reference_papers(reference_papers: list):
    summaries = {}
    for title in reference_papers:
        # 获取论文信息
        papers_data = await fetch_top_papers_for_keyword(title, per_type=1)
        paper_info = None
        if papers_data.get("most_relevant"):
            paper_info = papers_data["most_relevant"][0]
        elif papers_data.get("most_cited"):
            paper_info = papers_data["most_cited"][0]

        if not paper_info:
            print(f"⚠️ No paper found for title: {title}")
            continue
            # summaries[title] = {
            #     "title": title,
            #     "author": "未知作者",
            #     "year": "未知年份",
            #     "summary": "No paper information found."
            # }
            # continue

        summary_text = await summarize_single_paper_with_llm(paper_info)


        year = paper_info.get("publication_year") or paper_info.get("year") or "未知年份"
        
        doi = paper_info.get("doi")
        openalex_id = paper_info.get("id")
        link = doi if doi else openalex_id if openalex_id else "无链接"
        summaries[title] = {
            "title": paper_info.get("title", title),
            "author": paper_info.get("first_author", "未知作者"),
            "year": year,
            "summary": summary_text,
            "link": link
        }

    return summaries

async def summarize_all_papers(paper_infos):
    summaries = {}
    for kw, papers in paper_infos.items():
        summaries[kw] = {"most_cited": [], "most_relevant": []}
        for section in ["most_cited", "most_relevant"]:
            for paper in papers[section]:
                summary = await summarize_single_paper_with_llm(paper)
                
                year = paper.get("publication_year") or paper.get("year") or "未知年份"
                doi = paper.get("doi")
                openalex_id = paper.get("id")
                link = doi if doi else openalex_id if openalex_id else "无链接"
                summaries[kw][section].append({
                    "title": paper.get("title"),
                    "author": paper.get("first_author", "未知作者"),
                    "year": year,
                    "summary": summary,
                    "link": link
                })
    return summaries

def build_literature_review_prompt(
    keyword_summaries: dict,
    reference_summaries: dict = None
) -> str:
    prompt = "The following papers were retrieved based on keywords. Each entry includes the title, first author, publication year, and a brief summary. These papers may be useful for a literature review on the topic.\n\n"

    for kw, papers in keyword_summaries.items():
        prompt += f"Keyword: {kw}\n"
        for section in ["most_cited", "most_relevant"]:
            prompt += f"{section.capitalize()}:\n"
            for paper in papers[section]:
                prompt += (
                    f" - Title: {paper['title']}\n"
                    f"   Author: {paper['author']}\n"
                    f"   Year: {paper['year']}\n"
                    f"   Summary: {paper['summary']}\n"
                )
        prompt += "\n"

    if reference_summaries:
        prompt += "Reference papers suggested for literature review:\n"
        for title, paper in reference_summaries.items():
            prompt += (
                f" - Title: {paper['title']}\n"
                f"   Author: {paper['author']}\n"
                f"   Year: {paper['year']}\n"
                f"   Summary: {paper['summary']}\n"
            )
        prompt += "\n"
    References = "References:\n"
    ref_index = 1

    # Add references from keyword summaries
    for kw, papers in keyword_summaries.items():
        for section in ["most_cited", "most_relevant"]:
            for paper in papers[section]:
                References += (
                    f"{ref_index}. {paper['author']} ({paper['year']}). {paper['title']}. {paper['link']}\n"
                )
                ref_index += 1

    # Add references from recommendation list
    if reference_summaries:
        for title, paper in reference_summaries.items():
            References += (
                f"{ref_index}. {paper['author']} ({paper['year']}). {paper['title']}. {paper['link']}\n"
            )
            ref_index += 1
    prompt += (
        "Based on all the above papers, please generate a coherent, well-structured, and academically polished literature review. "
        "Organize the review using clear thematic structure, highlight major research threads, synthesize trends across papers, "
        "and provide insightful comparisons or contrasts where relevant. "
        "Emphasize depth, accuracy, and conceptual clarity. "
        "Do not invent paper information; summarize and integrate only what is provided. "
        "Write with the tone and rigor of a top-tier conference survey. "
        "Be approximately 3500 words long. "
        "Begin directly with a strong title or opening sentence, with no meta-introduction or phrases such as 'Here is' or 'Below is'. "
        "Ensure the writing reads naturally, with smooth transitions and cohesive narrative flow."
    )

    return prompt,References

async def get_literature_review_prompt(query: str, progress_callback):
    
    await progress_callback("Step 1/6: Identifying core concepts and extracting domain-specific keywords...")
    keywords = await get_keywords_from_model(query)
    
    await progress_callback("Step 2/6: Retrieving highly relevant papers based on semantic and citation signals...")
    paper_infos = await fetch_all_papers(keywords)

    await progress_callback("Step 3/6: Collecting foundational and influential reference works for contextual grounding...")
    reference_papers = await get_reference_papers(query)

    await progress_callback("Step 4/6: Synthesizing key insights from retrieved papers ...")
    summaries = await summarize_all_papers(paper_infos)

    await progress_callback("Step 5/6: Integrating synthesized insights into a coherent, comprehensive literature understanding ...")

    reference_summaries = await summarize_reference_papers(reference_papers)

    prompt = build_literature_review_prompt(summaries, reference_summaries)
    
    return prompt
@app.post("/literature_review")
async def literature_review(request: Request):
    """
    Literature review endpoint - uses standard LLM model

    Request body:
    {
        "query": "What are the latest advances in transformer models?"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        print(f"[literature_review] Received query: {query}")
        print(f"[literature_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        async def generate():

            # --- 用队列在 generate 和 callback 之间传递消息 ---
            queue = asyncio.Queue()

            # --- SSE helper ---
            def sse_text(text: str):
                return (
                    "data: "
                    + json.dumps({
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {"content": text}}]
                    })
                    + "\n\n"
                )

        
            async def progress_callback(msg: str):
                await queue.put(msg)
        
            async def run_prompt_job():
                prompt, References = await get_literature_review_prompt(query, progress_callback)
                await queue.put("[FINAL_PROMPT]" + prompt)
                await queue.put("[REFERENCES]" + References)
                await queue.put(None)   # 结束信号

            # 启动后台任务
            asyncio.create_task(run_prompt_job())

            # --- 主循环：从 queue 读消息并 SSE 输出 ---
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                
                # 收到最终 prompt → 开始调用模型并流式返回
                if msg.startswith("[FINAL_PROMPT]"):
                    full_prompt = (
                        "Conduct a literature review on the following topic: "
                        + query
                        + "\n"
                        + msg[len("[FINAL_PROMPT]"):]
                    )

                    # 调用大模型（流式）
                    # stream = await client.chat.completions.create(
                    #     model=os.getenv("SCI_LLM_MODEL"),
                    #     messages=[{"role": "user", "content": full_prompt}],
                    #     max_tokens=4096,
                    #     temperature=0.2,
                    #     stream=True
                    # )

                    # async for chunk in stream:
                    #     yield "data: " + json.dumps(chunk.model_dump()) + "\n\n"
                    # yield sse_text("Generating review draft...\n")

                    # yield sse_text("Generating literature review ...\n")
                    stream = await client.chat.completions.create(
                        model=os.getenv("SCI_LLM_MODEL"),
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=4096*2,
                        temperature=0.2,
                        stream=True
                    )
                    full_text = ""
                    async for chunk in stream:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta_content = chunk.choices[0].delta.content
                            if delta_content:
                                full_text += delta_content

                    new_prompt = (
                        "You are a High-Score Academic Polisher. Your task is to transform the following text into a version that "
                        "is significantly clearer, more coherent, more persuasive, and more academically polished—optimized for "
                        "human reviewers who value strong structure, precise articulation, and professional scholarly presentation.\n\n"
                        "Be approximately 3000 words long. "

                        "Your improvements should:\n"
                        "1) Strengthen logical flow, including explicit connections between ideas and a clear argumentative arc.\n"
                        "2) Increase clarity, precision, and readability while preserving a formal, academic tone.\n"
                        "3) Make claims sound more rigorous, well-supported, and confidently articulated.\n"
                        "4) Enhance transitions and ensure a smooth progression of concepts.\n"
                        "5) Remove redundancy, vague expressions, and awkward or informal phrasing.\n"
                        "6) Preserve all original meaning—do NOT add new facts or modify the scientific content.\n"
                        "7) Emphasize coherence, depth, and an impression of expertise that aligns with top-tier academic writing.\n"
                        "8) **Where appropriate, restructure parts of the content using concise and well-formatted Markdown tables** "
                        "to improve organization, highlight contrasts, summarize contributions, or clarify multi-part structures. "
                        "These tables must not introduce new information—they should only reformat existing content.\n\n"

                        "Do NOT introduce new content. Only polish, restructure, and clarify expression.\n"
                        "Begin directly with the improved, high-score version.\n\n"
                        "Here is the text:\n"
                        f"{full_text}"
                    )


                    yield sse_text("Step 6/6: Polishing review draft...\n")
                    stream = await client.chat.completions.create(
                        model=os.getenv("SCI_LLM_MODEL"),
                        messages=[{"role": "user", "content": new_prompt}],
                        max_tokens=4096*2,
                        temperature=0.2,
                        stream=True
                    )
                    full_text = ""
                    async for chunk in stream:
                        yield "data: " + json.dumps(chunk.model_dump()) + "\n\n"
                        if chunk.choices and len(chunk.choices) > 0:
                            delta_content = chunk.choices[0].delta.content
                            if delta_content:
                                full_text += delta_content

                else:
                    if msg.startswith("[REFERENCES]"):
                        references_text = msg[len("[REFERENCES]"):]
                        yield sse_text("\n\n" + references_text + "\n")
                    else:
                    # 普通进度消息
                        yield sse_text(msg + "\n")

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/paper_qa")
async def paper_qa(request: Request):
    """
    Paper Q&A endpoint - uses reasoning model with PDF content

    Request body:
    {
        "query": "Please carefully analyze and explain the reinforcement learning training methods used in this article.",
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        pdf_content = body.get("pdf_content", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        print(f"[paper_qa] Received query: {query}")
        print(f"[paper_qa] Using reasoning model: {os.getenv('SCI_LLM_REASONING_MODEL')}")

        async def generate():
            # Extract text from PDF
            text = extract_pdf_text_from_base64(pdf_content)

            # Build prompt with PDF content
            prompt = f"""
                You are a scientific reviewer. Answer the question based **only** on the content of the paper below. 
                Provide a structured, detailed, and technically accurate response. 
                
                Paper:
                {text}
                
                Question: {query}
                
                Instructions:
                - Make sure your answer demonstrates **understanding** of the paper’s content.
                - Go beyond mere restatement; provide reasoning, analysis, or critique (**depth**).
                - Directly address the question (**relevance**).
                - Only use information supported by the paper; do not hallucinate (**faithfulness**).
                - Structure your answer clearly, using headings or numbered points if helpful (**clarity**).
                """


            # Call reasoning model with streaming
            stream = await client.chat.completions.create(
                model=os.getenv("SCI_LLM_REASONING_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.2,
                stream=True
            )

            # Stream back results
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # Extract and log reasoning content
                    reasoning_content = getattr(delta, 'reasoning_content', None)
                    if reasoning_content:
                        print(f"[paper_qa] Reasoning: {reasoning_content}", flush=True)

                    # Stream regular content to client
                    delta_content = delta.content
                    if delta_content:
                        response_data = {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {
                                    "content": delta_content
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/ideation")
async def ideation(request: Request):
    """
    Ideation endpoint - uses embedding model for similarity and LLM for generation

    Request body:
    {
        "query": "Generate research ideas about climate change"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        print(f"[ideation] Received query: {query}")
        print(f"[ideation] Using model: {os.getenv('SCI_LLM_MODEL')}")

        async def generate():

            queue = asyncio.Queue()

            def sse_text(text: str):
                return (
                    "data: "
                    + json.dumps({
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {"content": text}}]
                    })
                    + "\n\n"
                )

            # --- 起始输出 ---
            # yield sse_text("Starting search and summary of relevant papers...It may take a few minutes.\n")
        
            async def progress_callback(msg: str):
                # await queue.put(msg)
                pass

            async def run_prompt_job():
                prompt, _ = await get_literature_review_prompt(query, progress_callback)
                await queue.put("[FINAL_PROMPT]" + prompt)
                await queue.put(None)   # 结束信号

            # 启动后台任务
            asyncio.create_task(run_prompt_job())

            # --- 主循环：从 queue 读消息并 SSE 输出 ---
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                
                # 收到最终 prompt → 开始调用模型并流式返回
                if msg.startswith("[FINAL_PROMPT]"):
                    full_prompt = (
                        "Conduct a literature review on the following topic: "
                        + query
                        + "\n"
                        + msg[len("[FINAL_PROMPT]"):]
                    )

                    stream = await client.chat.completions.create(
                        model=os.getenv("SCI_LLM_MODEL"),
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=4096,
                        temperature=0.2,
                        stream=True
                    )
                    full_text = ""
                    async for chunk in stream:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta_content = chunk.choices[0].delta.content
                            if delta_content:
                                full_text += delta_content

                    new_prompt = (
                        f"You are an expert researcher tasked with proposing genuinely novel, high-impact research ideas "
                        f"based strictly on the topic: {query}. Your ideas must go significantly beyond the literature review "
                        f"provided below, using it only as background context rather than something to summarize or rephrase.\n\n"
                        f"{full_text}\n\n"
                        "Generate entirely new, original research directions that would be considered innovative by top-tier "
                        "researchers. Do NOT summarize, repeat, or restate any part of the review.\n\n"
                        "Your output should:\n"
                        "• Focus exclusively on producing breakthrough research ideas related to the topic.\n"
                        "• Be scientifically rigorous, technically feasible, and clearly grounded in current knowledge.\n"
                        "• Show deep understanding by proposing ideas that are specific, actionable, and non-obvious.\n"
                        "• For each idea, explicitly include:\n"
                        "    1) What makes it novel compared to existing work,\n"
                        "    2) A concrete methodological plan or possible technical approaches,\n"
                        "    3) The potential scientific or practical impact.\n"
                        "• Avoid surface-level insights—aim for conceptual depth and multi-step reasoning.\n"
                        "• Deliver approximately 1500 words.\n"
                        "• Begin immediately with a strong, compelling title.\n"
                        "• Organize ideas clearly (e.g., numbered, categorized by conceptual difficulty, or arranged from foundational to ambitious).\n\n"
                         "Begin directly with a strong title or opening sentence, with no meta-introduction or phrases such as 'Here is' or 'Below is'. "
                        "Your goal is to deliver research ideas that a human evaluator would consider highly creative, "
                        "substantive, and publication-worthy."
                    )

                    
                    print(new_prompt)
                    stream = await client.chat.completions.create(
                        model=os.getenv("SCI_LLM_MODEL"),
                        messages=[{"role": "user", "content": new_prompt}],
                        max_tokens=4096,
                        temperature=0.2,
                        stream=True
                    )

                    async for chunk in stream:
                        yield "data: " + json.dumps(chunk.model_dump()) + "\n\n"
                else:
                    # 普通进度消息
                    yield sse_text(msg + "\n")

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )

@app.post("/paper_review")
async def paper_review(request: Request):
    """
    Paper review endpoint - uses LLM model with PDF content

    Request body:
    {
        "query": "Please review this paper",  # optional, default review prompt will be used
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "Please provide a comprehensive review of this paper")
        pdf_content = body.get("pdf_content", "")

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        print(f"[paper_review] Received query: {query}")
        print(f"[paper_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        async def generate():
            # Extract text from PDF
            text = extract_pdf_text_from_base64(pdf_content)

            # Build prompt with PDF content
            prompt = f"""
                You are an expert academic reviewer. 
                Your task is to provide a **structured review** of the following paper. 
                Do NOT summarize the text in a casual way—follow the exact format required by Paper Review:

                Paper:
                {text}

                Output format:

                1. Summary: Provide a concise summary of the paper.
                2. Strengths: List the key strengths and contributions.
                3. Weaknesses / Concerns: List important weaknesses, limitations, or concerns.
                4. Questions for Authors: List any clarifying questions or points that need author input.
                5. Scores:
                   - Overall (0-10): 
                   - Novelty (0-10): 
                   - Technical Quality (0-10): 
                   - Clarity (0-10): 
                   - Confidence (0-5): 

                Instructions:
                - Provide **professional, detailed, and specific feedback** for each section.
                - Scores should be **consistent with your textual reasoning**.
                - Be balanced: discuss both strengths and weaknesses fairly.
                - Ensure clarity and readability, with well-structured, organized text.
                - Do not include any personal opinions outside the context of reviewing the paper.
                """


            # Call LLM model with streaming
            stream = await client.chat.completions.create(
                model=os.getenv("SCI_LLM_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.2,
                stream=True
            )

            # Stream back results
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        response_data = {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {
                                    "content": delta_content
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
