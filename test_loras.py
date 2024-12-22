print('___________________Testing / Inference phase __________________')

unet_lora.eval()
text_encoder_lora.eval()
clear_cache()
new_pipe = StableDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=trainer.best_text_encoder.to(device, dtype=WEIGHT_DTYPE),
    vae=vae,
    unet=trainer.best_unet.to(device, dtype=WEIGHT_DTYPE),
    scheduler=noise_scheduler,
    safety_checker= None,
    feature_extractor=None
)

new_pipe.to(device)

check_and_make_folder(f'{save_result_path}/{folder_name}')
check_and_make_folder(f'{save_result_path}/{folder_name}/test_images')

for place in range(len(test_df)):
    temp_prompts = test_df.text.iloc[place]

    temp = new_pipe(temp_prompts, height = 224, width = 224).images[0]

    temp.save(f'{save_result_path}/{folder_name}/test_images/{place}.png')

    display(temp)