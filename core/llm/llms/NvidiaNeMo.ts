import { BaseLLM } from "..";
import {
  ChatMessage,
  CompletionOptions,
  LLMOptions,
  ModelProvider,
} from "../..";
import { stripImages } from "../countTokens";
import { streamResponse } from "../stream";

class NvidiaNeMo extends BaseLLM {
  static providerName: ModelProvider = "nvidia-nemo";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.llm.ngc.nvidia.com/v1/models",
    model: "gpt-43b-905",
  };

  private _convertMessage(message: ChatMessage) {
    if (typeof message.content === "string") {
      return message;
    }

    return {
      role: message.role,
      content: stripImages(message.content),
      images: message.content
        .filter((part) => part.type === "imageUrl")
        .map((part) => part.imageUrl?.url.split(",").at(-1)),
    };
  }

  private _convertArgs(
    options: CompletionOptions,
    prompt: string | ChatMessage[]
  ) {
    const finalOptions: any = {
      temperature: options.temperature,
      top_p: options.topP,
      top_k: options.topK,
      tokens_to_generate: options.maxTokens,
      stop: options.stop,
    };

    if (typeof prompt === "string") {
      finalOptions.chat_context = [
        {
          role: "user",
          content: prompt,
        },
      ];
    } else {
      finalOptions.chat_context = prompt.map(this._convertMessage);
    }

    return finalOptions;
  }

  protected async *_streamChat(
    messages: string | ChatMessage[],
    options: CompletionOptions
  ): AsyncGenerator<ChatMessage> {
    const response = await this.fetch(`${this.apiBase}/${this.model}/chat`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        Accept: "text/event-stream",
        "Content-Type": "application/json",
        "x-stream": "true"
      },
      body: JSON.stringify(this._convertArgs(options, messages)),
    });

    let buffer = "";
    for await (const value of streamResponse(response)) {
      // Append the received chunk to the buffer
      buffer += value;
      // Split the buffer into individual JSON chunks
      const chunks = buffer.split("\n");
      buffer = chunks.pop() ?? "";

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        if (chunk !== "") {
          try {
            const j = JSON.parse(chunk);
            if (
              j?.text !== undefined
            ) {
              yield {
                role: "assistant",
                content: j.text
              }
            } else if (j.error) {
              throw new Error(`Error: ${j.error}`);
            } else {
              throw new Error(`Unknown error`);
            }
          } catch (e) {
            throw new Error(`Error parsing NeMo response: ${e} ${chunk}`);
          }
        }
      }
    }
  }

  protected async *_streamComplete(
    prompt: string,
    options: CompletionOptions
  ): AsyncGenerator<string> {
    for await (const chunk of this._streamChat(prompt, options)) {
      yield stripImages(chunk.content);
    }
  }
}

export default NvidiaNeMo;
