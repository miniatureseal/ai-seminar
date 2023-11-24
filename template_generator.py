from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate


class TemplateGenerator:
    _PROMPT_TEMPLATE = "{introduction}\n\n" + "{context_placeholder}\n\n" + "{start}"

    _PROMPT_INTRODUCTION = "Your job is it to produce possible chat replies for the user {activeUserId} in the data which is provided below.\n"

    _PROMPT_START = (
        "Here is the chat so far:"
        "------------\n"
        "{chat_history}\n"
        "------------\n"
        "Here are messages which might be relevant in writing a reply to the other user:"
        "------------\n"
        "{previous_chat_context}\n"
        "------------\n"
        "Now, your task is to generate three possible replies which user {activeUserId} could write taking into consideration the chat and the provided profile for the user.\n"
        "Take into consideration the users writing style so far.\n"
        "If you can't write a chat message because you are missing information which the user itself might know then don't generate any possible reply."
    )

    _CONTEXT_INFO_COMPANY = "User {activeUserId} is a representant of a company and is seeking people to hire for a job."

    _CONTEXT_INFO_JOB_SEEKER = "User {activeUserId} is a job seeker and is writing with a representant from a company which is offers a job he is interested in."

    def __init__(self):
        introduction_prompt = PromptTemplate.from_template(self._PROMPT_INTRODUCTION)
        context_info_user_prompt = PromptTemplate.from_template(
            self._CONTEXT_INFO_JOB_SEEKER
        )
        context_info_company_prompt = PromptTemplate.from_template(
            self._CONTEXT_INFO_COMPANY
        )
        start_prompt = PromptTemplate.from_template(self._PROMPT_START)
        self.full_template = PromptTemplate.from_template(self._PROMPT_TEMPLATE)
        self.job_seeker_prompt_template = [
            ("introduction", introduction_prompt),
            ("context_placeholder", context_info_user_prompt),
            ("start", start_prompt),
        ]
        self.company_prompt_template = [
            ("introduction", introduction_prompt),
            ("context_placeholder", context_info_company_prompt),
            ("start", start_prompt),
        ]

    def get_company_prompt_template(self):
        return PipelinePromptTemplate(
            final_prompt=self.full_template,
            pipeline_prompts=self.company_prompt_template,
        )

    def get_job_seeker_prompt_template(self):
        return PipelinePromptTemplate(
            final_prompt=self.full_template,
            pipeline_prompts=self.job_seeker_prompt_template,
        )
