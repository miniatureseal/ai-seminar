from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate


class TemplateGenerator:
    _PROMPT_TEMPLATE = (
        "{introduction}\n" + "{context_placeholder}\n" + "{start}\n" + "{format}"
    )

    _PROMPT_INTRODUCTION = "Please generate three possible chat replies for user {active_user_id}, taking into account the chat so far and especially focusing on incorporating the knowledge from the context.\n"

    _CONTEXT_INFO_COMPANY = "User {active_user_id} is a representative of a company and is seeking people to hire for a job. The context includes past interactions and discussions that occurred between the company {active_user_id} and potential job seekers. The replies should address all questions and open points of the latest message from the job seeker.\n"

    _CONTEXT_INFO_JOB_SEEKER = "User {active_user_id} is a job seeker and is writing with a representative from a company which offers a job he is interested in. The context includes past interactions and discussions that occurred between the job seeker {active_user_id} and companies he is interested in. The replies should address all questions and open points of the latest message from the company representative.\n"

    _PROMPT_START = (
        "The replies should only include relevant information known through the context messages from previous chats and is needed to answer the text. Avoid making up new information and don't deviate from the user's writing style. Also, please ensure that all replies convey the same information but are rephrased differently. Don't greet the user which you are replying to again if you already greeted him in the chat history.\n\n"
        "Context:"
        "------------\n"
        "{previous_chat_context}\n"
        "------------\n"
        "Chat history:"
        "------------\n"
        "{chat_history}\n"
        "------------\n"
        "Latest message from user {chat_partner_user_id} to reply to:"
        "------------\n"
        "{message_to_reply_to}\n"
        "------------\n"
    )

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
