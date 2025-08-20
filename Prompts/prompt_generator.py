
from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    template="""
please summarize the research paper titled "{paper_input}" with the following specification:
explanation style: {style_input}
explanation length: {length_input}
1. Mathematical Details:
-include relevant mathematical equations if present in the paper.
-Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
-use relatable analogies to simplify complex ideas.
if certain information is not available in the paper, respond with:"insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.

""",
input_variables=['paper_input','style_input','length_input'],
validate_template=True


)

template.save('template.json')