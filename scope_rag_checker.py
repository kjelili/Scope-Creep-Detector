import openai
import json

def check_scope_creep_with_rag(email, scope_text, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant helping a project manager detect scope creep. "
                    "You will receive the current project scope and an email. Your task is to:\n"
                    "1. Determine if the email contains potential scope creep (yes/no).\n"
                    "2. Justify your reasoning.\n"
                    "3. Suggest an appropriate response.\n"
                    "4. Rate the risk level using the standard risk matrix based on impact and likelihood (Low, Moderate, High, Extreme).\n"
                    "5. Using Retrieval-Augmented Generation (RAG), analyze an incoming email to identify potential scope changes in a project. "
                    "Based on the identified changes, assess the likely impact on project cost and timeline. "
                    "The analysis should extract key information from the email relevant to project deliverables, requirements, or expectations, "
                    "compare this information against an established project scope (e.g., project charter, statement of work, previous agreements) provided as part of the retrieval corpus, "
                    "flag any discrepancies that indicate a deviation from the agreed-upon scope. "
                    "For each flagged scope change, estimate the potential impact on project cost and timeline. "
                    "Provide a concise summary of the identified scope changes and their estimated cost and timeline impacts, suitable for a project manager or stakeholder.\n"
                    "Respond in JSON format with keys: scope_creep, justification, suggestion, risk_level, reference_scope_line, impact_analysis."
                )
            },
            {
                "role": "user",
                "content": f"Project Scope:\n{scope_text}\n\nEmail:\n{email}"
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        return {
            "scope_creep": "error",
            "justification": str(e),
            "suggestion": "check logs",
            "risk_level": "Unknown",
            "reference_scope_line": "none",
            "impact_analysis": "none"
        }
