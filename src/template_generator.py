from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate


class TemplateGenerator:
    _PROMPT_TEMPLATE = (
        "{introduction}\n\n" + "{context_placeholder}\n\n" + "{start}\n\n" + "{format}"
    )

    _PROMPT_INTRODUCTION = "Your job is it to produce possible chat replies for the user {active_user_id} while taking into consideration additonal data which is provided below.\n"

    _PROMPT_START = (
        "Relevant messages from previous chats:"
        "------------\n"
        "{previous_chat_context}\n"
        "------------\n"
        "The chat so far:"
        "------------\n"
        "{chat_history}\n"
        "------------\n"
        "Generate three possible replies user {active_user_id} could write taking into consideration the chat so far, as well as the relevant messages provided from previous chats.\n"
        "Take into consideration the users writing style so far. Each generated reply should address all questions and open points of the opposing parties latest message of user {chat_partner_user_id} given the timestamp of the message. The replies should not include information already written in the chat so far. Keep the answer short.\n\n"
    )

    _CONTEXT_INFO_COMPANY = "User {active_user_id} is a representant of a company and is seeking people to hire for a job.\n"

    _CONTEXT_INFO_JOB_SEEKER = "User {active_user_id} is a job seeker and is writing with a representant from a company which is offers a job he is interested in.\n"

    _FORMAT_INSTRUCTIONS = "{format_instructions}"

    def __init__(self, output_parser):
        introduction_prompt = PromptTemplate.from_template(self._PROMPT_INTRODUCTION)
        context_info_user_prompt = PromptTemplate.from_template(
            self._CONTEXT_INFO_JOB_SEEKER
        )
        context_info_company_prompt = PromptTemplate.from_template(
            self._CONTEXT_INFO_COMPANY
        )
        start_prompt = PromptTemplate.from_template(self._PROMPT_START)
        format_prompt = PromptTemplate.from_template(self._FORMAT_INSTRUCTIONS)
        format_prompt = format_prompt.partial(
            format_instructions=output_parser.get_format_instructions()
        )

        self.full_template = PromptTemplate.from_template(self._PROMPT_TEMPLATE)

        self.job_seeker_prompt_template = [
            ("introduction", introduction_prompt),
            ("context_placeholder", context_info_user_prompt),
            ("start", start_prompt),
            ("format", format_prompt),
        ]
        self.company_prompt_template = [
            ("introduction", introduction_prompt),
            ("context_placeholder", context_info_company_prompt),
            ("start", start_prompt),
            ("format", format_prompt),
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
