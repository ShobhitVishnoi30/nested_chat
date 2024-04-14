from autogen.agentchat import (
    GroupChat,
    AssistantAgent,
    UserProxyAgent,
    GroupChatManager,
)
from tools import url_scraper_tool, generate_function_config


config_list = [
    {
        "model": "gpt-4",
        "api_key": "",
        # put your api key here
    },
]


gpt_config = {
    "cache_seed": None,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 100,
}

confi_without_function = {
    "cache_seed": None,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 100,
}


gpt_config["functions"] = [generate_function_config(url_scraper_tool)]

user_agent_task = f"""You are excellent instruction follower which follows all the instruction very accurately and never deviates from those
You are helful assistant which is helping to get the health information and recipe information."""


user_proxy = UserProxyAgent(
    name="User",
    system_message=user_agent_task,
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
    human_input_mode="TERMINATE",
    llm_config=gpt_config,
    description="""You are working as a user proxy and deciding what agent should be selected according to user's input.
""",
)


user_proxy_with_no_auto_reply = UserProxyAgent(
    name="UserNoAutoReply",
    system_message=user_agent_task,
    max_consecutive_auto_reply=0,
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
    human_input_mode="TERMINATE",
    llm_config=gpt_config,
    description="""You are working as a user proxy and deciding what agent should be selected according to user's input.
""",
)


user_proxy.register_function(
    function_map={
        url_scraper_tool.name: url_scraper_tool._run,
    }
)

recipe_parser_task=f"""
You are responsible for getting the recipe PDF from the user and providing a detailed summary of it. Follow these instructions:

1. Always remember to attach `TERMINATE` at the end of your message when the user should provide some input.
2. Do not deviate from the given instructions and do not talk about irrelevant topics.
3. Your first task is to ask the user for the public drive URL of the recipe PDF.
4. If the user asks about the format of the recipe PDF, explain that it should be a PDF document containing the recipe's ingredients, instructions, and nutritional information.
5. After the user has provided the recipe PDF document, your second task is to verify if it is a valid recipe PDF. Check if it contains the necessary sections such as ingredients, instructions, and nutritional information.
6. If the provided document does not contain the specified details, ask the user to provide the correct document that includes those details.
7. If the document is valid, thank the user and provide a detailed summary of the recipe PDF, including the ingredients, instructions, and nutritional information, with proper headings and paragraphs.
8. Keep the first instruction in mind and attach `TERMINATE` at the end of your message when you need the user to provide input.
"""

health_report_parser_task = f"""
You are responsible for getting the health report PDF from the user and extracting relevant information about their current health condition, dietary restrictions, and allergies. Follow these instructions:

1. Always remember to attach `TERMINATE` at the end of your message when the user should provide some input.
2. Do not deviate from the given instructions and do not talk about irrelevant topics.
3. Your first task is to ask the user for the public drive URL of their health report PDF.
4. If the user asks about the format of the health report PDF, explain that it should be a PDF document containing information about their current health condition, dietary restrictions, and allergies.
5. After the user has provided the health report PDF document, your second task is to verify if it is a valid health report PDF. Check if it contains the necessary sections such as health condition, dietary restrictions, and allergies.
6. If the provided document does not contain the specified details, ask the user to provide the correct document that includes those details.
7. If the document is valid, thank the user and extract the relevant information about their current health condition, dietary restrictions, and allergies from the PDF. Provide this information in a well-structured format with proper headings and paragraphs.
8. Keep the first instruction in mind and attach `TERMINATE` at the end of your message when you need the user to provide input.
"""

health_report_updater_task = f"""
You are responsible for updating the health report information based on user input. Follow these instructions:

1. Always remember to attach `TERMINATE` at the end of your message when the user should provide some input.
2. Do not deviate from the given instructions and do not talk about irrelevant topics.
3. Your first task is to ask the user if they want to update any information in their health report.
4. If the user wants to update the information, ask them to provide the specific details they want to update, such as changes in their health condition, dietary restrictions, or allergies.
5. Update the existing health report information with the user's input, ensuring that you incorporate all the changes accurately.
6. After updating the information, provide the full and updated health report information in a well-structured format with proper headings and paragraphs.
7. If the user does not want to update the information, provide the existing health report information as it is.
8. Keep the first instruction in mind and attach `TERMINATE` at the end of your message when you need the user to provide input.
"""

recipe_suitability_analyzer_task = f"""
You are responsible for analyzing if the recipe is suitable for the user's health condition, dietary restrictions, and allergies based on the parsed information. Follow these instructions:

1. Always remember to attach `TERMINATE` at the end of your message when the user should provide some input.
2. Do not deviate from the given instructions and do not talk about irrelevant topics.
3. Your task is to analyze the recipe's ingredients, instructions, and nutritional information against the user's health report information, which includes their current health condition, dietary restrictions, and allergies.
4. Perform a thorough analysis to determine if the recipe is suitable for the user's health condition, considering any potential adverse effects or contraindications.
5. Check if the recipe's ingredients contain any items that conflict with the user's dietary restrictions or allergies.
6. Based on your analysis, provide a detailed assessment of the recipe's suitability for the user. If the recipe is suitable, clearly state that it is a suitable option and explain why. If the recipe is not suitable, clearly state that it is not a suitable option and explain the reasons, such as specific ingredients or nutritional factors that may be problematic for the user's health condition or dietary restrictions.
7. Keep in mind that you should not make any recommendations or suggest alternatives at this stage. Your task is solely to analyze and assess the suitability of the provided recipe based on the available information.
8. Attach `TERMINATE` at the end of your message if you need any additional information from the user to complete your analysis.
"""

recipe_parser = AssistantAgent(
    name="Recipe_parser",
    system_message=recipe_parser_task,
    llm_config=gpt_config,
    description=f"""You are the agent responsible for getting the recipe pdf.
    Get The pdf and download it.
""",
)


health_report_parser = AssistantAgent(
    name="Health_Report_Parser",
    system_message=health_report_parser_task,
    llm_config=gpt_config,
    description=f"""You are the agent responsible for getting the user's health report PDF and extracting relevant information about their health condition, dietary restrictions, and allergies.""",
)


health_report_updater = AssistantAgent(
    name="Health_Report_Updater",
    system_message=health_report_updater_task,
    llm_config=confi_without_function,
    description=f"""You are the agent responsible for updating the user's health report information based on their input.""",
)


recipe_suitability_analyzer = AssistantAgent(
    name="Recipe_Suitability_Analyzer",
    system_message=recipe_suitability_analyzer_task,
    llm_config=confi_without_function,
    description=f"""You are the agent responsible for analyzing if the recipe is suitable for the user's health condition, dietary restrictions, and allergies based on the parsed information.""",
)


recipe_summariser = {
    "summary_prompt": "You will be given a conversation where the user provides a recipe PDF and it parse the PDF and extract the ingredients, instructions, and nutritional information. Provide this information in detail and with proper headings and paragraphs. Do not add irrelevant information."
}

health_report_summariser = {
    "summary_prompt": f"""You will be given a conversation where the user provides a health report PDF and
      it parses the PDF and extract relevant information about the user's current health condition, dietary restrictions, and allergies.
      Provide this information in detail and with proper headings and paragraphs. Do not add irrelevant information."""
}

health_report_updater_summariser = {
    "summary_prompt": f"""YYou will given conversation in which user is updating the health report.
    Return final updated health report in detail and with proper headings and paragraphs.Do not add irrelevant information.
    If user is not updating the health report then return health report as it is."""
}

suitability_summariser = {
    "summary_prompt": f"""You will be given a conversation between the user and agents, along with the parsed recipe information and the user's health report information in which it is analysing the recipe for the given health condition.
    Provide a detailed summary of the analysis and conversation.
    """
}


chat_results = user_proxy.initiate_chats(
    [
        {
            "recipient": recipe_parser,
            "message": "Ask for recipe and Parse the recipe PDF and extract the ingredients, instructions, and nutritional information",
            "max_turns": 5,
            "summary_method": "reflection_with_llm",
            "summary_args": recipe_summariser,
        },
        {
            "recipient": health_report_parser,
            "message": "Ask for health report and parse the health report PDF and extract relevant information about the user's current health condition, dietary restrictions, and allergies",
            "max_turns": 6,
            "summary_method": "reflection_with_llm",
            "summary_args": health_report_summariser,
            "finished_chat_summary": False,
            # It does not need summary of the previous chat so finished_chat_summary has been set to false
        },
        {
            "sender": user_proxy_with_no_auto_reply,
            "recipient": health_report_updater,
            "message": "Update the health report information according to user input. Always provide the full health report information in the response",
            "max_turns": 6,
            "summary_method": "reflection_with_llm",
            "summary_args": health_report_updater_summariser,
            "carryover_indexes": [1],
            # It only need summary from the second chat so putting 1 chat index here.
        },
        {
            "sender": user_proxy_with_no_auto_reply,
            "recipient": recipe_suitability_analyzer,
            "message": "Analyze if the recipe is suitable for the user's health condition, dietary restrictions, and allergies based on the parsed information",
            "max_turns": 4,
            "summary_method": "reflection_with_llm",
            "summary_args": suitability_summariser,
            "carryover_indexes": [0, 2],
            # It needs summary from the first and third chat so putting 0,2 chat index here.
        },
    ]
)
