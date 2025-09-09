prompts = {
    "sql": {
            "system_message":"""You are a text to SQL query translator. 
        Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.""",
            "user_prompt": """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""
    },
    "product": {
            "system_message": "You are an expert product description writer for Amazon.",
            "user_prompt": """Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

<PRODUCT>
{product}
</PRODUCT>

<CATEGORY>
{category}
</CATEGORY>
"""
    },
    "en-ko-trans": {
            "system_message": "You are an expert in translating English to Korean.",
            "user_prompt": """### Instruction:
Translate the following text from English to Korean.

### Input:
{sentence}
 
### Response:
"""
    }
}