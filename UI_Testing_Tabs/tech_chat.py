import gradio as gr

SYSTEM_CONTENT = "You are a DM for tabletop RPGs named Technomancer and are friendly. Assume the user already knows a lot of the terminology. You are not meant to generate new campaign ideas, rules, but are meant to help reference rules, tables, pages, NPCs, and the like."
SYSTEM_MODELS = ["phi4"]
RULE_SYSTEMS = ["AD&D", "SR3", "Rifts"]

with gr.Blocks() as chat:
    with gr.Row():
        chatbot = gr.Chatbot(type = "messages")

    with gr.Row():
        with gr.Column(scale = 12):
            msg = gr.Textbox(show_label = True, label = "Msg for Technomancer", submit_btn = True)
        with gr.Column(scale = 1):
            with gr.Row():
                save_btn = gr.Button(value = "Save Chat to DBoH (not implimented yet)")
            with gr.Row():
                clear_tn = gr.ClearButton([msg, chatbot], value = "Clear Chat")

    with gr.Row():
        with gr.Accordion(label = "Documents", open = False):
            gr.Markdown("To add or remove a document, go to Upload Tab.")
            with gr.Row():
                with gr.Column():
                    available_collections = gr.Dropdown(choices=[x for x in range(10)], info = "Collections of Rule systems", interactive = True)
                with gr.Column():
                    available_documents = gr.Dropdown(choices = [x for x in range(20)], info = "Available documents in chosen collection", interactive = True)

    with gr.Row():
        with gr.Accordion(label = "Advanced Features", open = False):
            gr.Markdown("Note, changing some of these features mid conversation might cause confusion/hallucinations within the chatbot.")
            active_prompt_display = gr.Textbox(label = "Current System State", value = SYSTEM_CONTENT, visible = False, interactive = False)
            change_sys_prompt = gr.Textbox(show_label = True, placeholder = SYSTEM_CONTENT, label = "Change System Prompt: Press enter to update", submit_btn = True)
            model_choice = gr.Dropdown(choices = SYSTEM_MODELS, value = SYSTEM_MODELS[0], label = "Model Choice", info = "This will report both Language models and Embedding models: please know which one is which when selecting here! You want the Language model.", interactive = True)
            # Manual RAG control
            with gr.Row():
                use_rag_toggle = gr.Checkbox(label = "Look at Rulebooks & Notes", value = False, info = "When enabled, Technomancer will search the available rule systems for answers.")
                collection_choice = gr.Dropdown(choices = RULE_SYSTEMS, value = RULE_SYSTEMS[0] if RULE_SYSTEMS else None, label = "Rule System", interactive = True)


if __name__ in "__main__":
    chat.launch()