import { BaseLLM } from "..";
import {
  ChatMessage,
  CompletionOptions,
  LLMOptions,
  ModelProvider,
} from "../..";
import { stripImages } from "../countTokens";
import { streamResponse } from "../stream";

const NVAIFM_MODELS: Record<string, string> = {
  "codellama-70b": "2ae529dc-f728-4a46-9b8d-2697213666d8",
  "llama2-70b": "0e349b44-440a-44e1-93e9-abe8dcb27158",
  "mistral-7b": "35ec3354-2681-4d0e-a8dd-80325dcf7c63",
  "mistral-8x7b": "8f4118ba-60a8-4e6b-8574-e38a4067a4a3",
};

class NvidiaAIFM extends BaseLLM {
  static providerName: ModelProvider = "nvidia-aifm";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions",
    model: "codellama-70b",
  };

  private _getModel() {
    const model = NVAIFM_MODELS[this.model];
    if (!model) throw new Error(`Unknown model: ${this.model}`);
    return model;
  }

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
      // The sampling temperature to use for text generation. The higher the temperature value is, the less deterministic the output text will be. It is not recommended to modify both temperature and top_p in the same call.
      temperature: options.temperature,
      // The top-p sampling mass used for text generation. The top-p value determines the probability mass that is sampled at sampling time. For example, if top_p = 0.2, only the most likely tokens (summing to 0.2 cumulative probability) will be sampled. It is not recommended to modify both temperature and top_p in the same call.
      top_p: options.topP,
      // The maximum number of tokens to generate in any given call. Note that the model is not aware of this value, and generation will simply stop at the number of tokens specified.
      max_tokens: options.maxTokens,
      // If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result.
      // seed: ?
      // A string or a list of strings where the API will stop generating further tokens. The returned text will not contain the stop sequence.
      stop: options.stop,
      // If set, partial message deltas will be sent. Tokens will be sent as data-only server-sent events (SSE) as they become available (JSON responses are prefixed by `data: `), with the stream terminated by a `data: [DONE]` message.
      stream: true,
    };

    if (typeof prompt === "string") {
      finalOptions.messages = [
        {
          role: "user",
          content: prompt,
        },
      ];
    } else {
      finalOptions.messages = prompt.map(this._convertMessage);
    }

    return finalOptions;
  }

  protected async *_streamChat(
    messages: string | ChatMessage[],
    options: CompletionOptions
  ): AsyncGenerator<ChatMessage> {
    const modelUUID = this._getModel();
    const response = await this.fetch(`${this.apiBase}/${modelUUID}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        Accept: "text/event-stream",
        "Content-Type": "application/json",
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
        // remove the `data: ` prefix from chunk
        const chunk = chunks[i].substring(6).trim();
        if (chunk !== "" && chunk !== "[DONE]") {
          try {
            const j = JSON.parse(chunk);
            if (
              j?.choices.length &&
              j.choices[0]?.delta?.content !== undefined
            ) {
              yield {
                role: j.choices[0].delta?.role || "assistant",
                content: j.choices[0].delta.content,
              };
            } else if (j.error) {
              throw new Error(`Unknown error: ${j.error}`);
            } else {
              throw new Error(`Unknown error`);
            }
          } catch (e) {
            throw new Error(`Error parsing NVAIFM response: ${e} ${chunk}`);
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

export default NvidiaAIFM;
