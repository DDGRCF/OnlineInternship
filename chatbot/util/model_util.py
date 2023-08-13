import os

from enum import Enum
from threading import Thread
from typing import Any, Iterator
from abc import ABC, abstractclassmethod

from loguru import logger


class ModelFactory:
    _MODELS_ = {} 

    @classmethod
    def get(cls, model_type: str, *args, **kwargs):
        MODEL = ModelType.get_model(model_type)
        if MODEL in ModelFactory._MODELS_:
            return ModelFactory._MODELS_[MODEL]
        else:
            model = MODEL(*args, **kwargs)
            ModelFactory._MODELS_[MODEL] = model
            return model

    @classmethod
    def pop(cls, model_type: str):
        MODEL = ModelType.get_model(model_type)
        ModelFactory._MODELS_.pop(MODEL, None)

    @classmethod
    def has(cls, model_type: str):
        MODEL = ModelType.get_model(model_type)
        return MODEL in ModelFactory._MODELS_


class Model(ABC): 
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    def do(
        self,
        message: str,
        history: list[tuple[str, str]], # figure out: why history_with_input
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        *args,
        **kwargs,
    ) -> Iterator[list[tuple[str, str]]]:
        assert max_new_tokens <= self.max_tokens, f"input tokens > model max tokens allow ({self.model.max_tokens})"

        generator = self.run(message, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
        # judge first run success!
        try:
            first_response = next(generator)
            yield first_response
        except StopIteration:
            logger.warning("error happened when do inference!")
            yield ""

        for response in generator:
            yield response


    def check_input_token_length(message: str, 
                                 chat_history: list[tuple[str, str]], 
                                 system_prompt: str) -> bool:
        len =  self.get_input_token_length(message, chat_history, system_prompt)
        return len < self.model.max_tokens

    def reduce_input_token(message: str, 
                           chat_history: list[tuple[str, str]], 
                           system_prompt: str) -> bool:
        return True

    @abstractclassmethod
    def run(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> Iterator[str]:
        pass

    @staticmethod
    @abstractclassmethod
    def get_prompt(message: str, chat_history: list[tuple[str, str]] = [], system_prompt: str = "") -> str:
        pass

    @abstractclassmethod
    def get_input_token_length(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
    ) -> int:
        pass


class LLaMA2(Model):
    def __init__(
        self,
        model_path: str = "",
        backend_type: str = "llama.cpp",
        max_tokens: int = 4012,
        load_in_8bit: bool = True,
        local_dir = "./models/",
        verbose: bool = False,
    ):
        """Load a llama2 model from `model_path`.

        Args:
            model_path: Path to the model.
            backend_type: Backend for llama2, options: llama.cpp, gptq, transformers
            max_tokens: Maximum context size.
            load_in_8bit: Use bitsandbytes to run model in 8 bit mode (only for transformers models).
            verbose: Print verbose output to stderr.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A LLaMA2 Wrapper instance
        """
        super().__init__(max_tokens=max_tokens)
        self.model_path = model_path
        self.backend_type = BackendType.get_type(backend_type)
        self.load_in_8bit = load_in_8bit

        self.model = None
        self.tokenizer = None

        self.verbose = verbose

        if self.backend_type is BackendType.LLAMA_CPP:
            logger.info("Running on backend llama.cpp.")
        else:
            import torch

            if torch.cuda.is_available():
                logger.info("Running on GPU with backend torch transformers.")
            else:
                logger.info("GPU CUDA not found.")

        self.default_llamacpp_path = f"{local_dir}/llama-2-7b-chat.ggmlv3.q4_0.bin"
        self.default_gptq_path = f"{local_dir}/Llama-2-7b-Chat-GPTQ"
        # Download default ggml/gptq model
        if self.model_path == "":
            logger.info("Model path is empty.")
            if self.backend_type is BackendType.LLAMA_CPP:
                logger.info("Use default llama.cpp model path: " + self.default_llamacpp_path)
                if not os.path.exists(self.default_llamacpp_path):
                    logger.info("Start downloading model to: " + self.default_llamacpp_path)
                    from huggingface_hub import hf_hub_download

                    hf_hub_download(
                        repo_id="TheBloke/Llama-2-7B-Chat-GGML",
                        filename="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        local_dir=local_dir,
                    )
                else:
                    logger.info(f"Model exists in {local_dir}llama-2-7b-chat.ggmlv3.q4_0.bin.")
                self.model_path = self.default_llamacpp_path
            elif self.backend_type is BackendType.GPTQ:
                logger.info("Use default gptq model path: " + self.default_gptq_path)
                if not os.path.exists(self.default_gptq_path):
                    logger.info("Start downloading model to: " + self.default_gptq_path)
                    from huggingface_hub import snapshot_download

                    snapshot_download(
                        "TheBloke/Llama-2-7b-Chat-GPTQ",
                        local_dir=self.default_gptq_path,
                    )
                else:
                    logger.info("Model exists in " + self.default_gptq_path)
                self.model_path = self.default_gptq_path

        self.init_tokenizer()
        self.init_model()

    def init_model(self):
        if self.model is None:
            self.model = LLaMA2.create_llama2_model(
                self.model_path,
                self.backend_type,
                self.max_tokens,
                self.load_in_8bit,
                self.verbose,
            )
        if self.backend_type is not BackendType.LLAMA_CPP:
            self.model.eval()

    def init_tokenizer(self):
        if self.backend_type is not BackendType.LLAMA_CPP:
            if self.tokenizer is None:
                self.tokenizer = LLaMA2.create_llama2_tokenizer(self.model_path)


    @staticmethod
    def get_prompt(
        message: str, chat_history: list[tuple[str, str]] = [], system_prompt: str = ""
    ) -> str:
        """Process message to llama2 prompt with with chat history
        and system_prompt for chatbot.

        Examples:
            >>> prompt = get_prompt("Hi do you know Pytorch?")

        Args:
            message: The origianl chat message to generate text from.
            chat_history: Chat history list from chatbot.
            system_prompt: System prompt for chatbot.

        Yields:
            prompt string.
        """
        texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
        for user_input, response in chat_history:
            texts.append(f"{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ")
        texts.append(f"{message.strip()} [/INST]")
        return "".join(texts)

    @classmethod
    def create_llama2_model(
        cls, model_path, backend_type, max_tokens, load_in_8bit, verbose
    ):
        if backend_type is BackendType.LLAMA_CPP:
            from llama_cpp import Llama

            model = Llama(
                model_path=model_path,
                n_ctx=max_tokens,
                n_batch=max_tokens,
                verbose=verbose,
            )
        elif backend_type is BackendType.GPTQ:
            from auto_gptq import AutoGPTQForCausalLM

            model = AutoGPTQForCausalLM.from_quantized(
                model_path,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
        elif backend_type is BackendType.TRANSFORMERS:
            import torch
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=load_in_8bit,
            )
        else:
            logger.info(backend_type + "not implemented.")
        return model

    @classmethod
    def create_llama2_tokenizer(cls, model_path):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer

    def get_token_length(
        self,
        prompt: str,
    ) -> int:
        if self.backend_type is BackendType.LLAMA_CPP:
            input_ids = self.model.tokenize(bytes(prompt, "utf-8"))
            return len(input_ids)
        else:
            input_ids = self.tokenizer([prompt], return_tensors="np")["input_ids"]
            return input_ids.shape[-1]

    def get_input_token_length(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
    ) -> int:
        prompt = LLaMA2.get_prompt(message, chat_history, system_prompt)

        return self.get_token_length(prompt)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Create a generator of response from a prompt.

        Examples:
            >>> llama2_wrapper = LLaMA2()
            >>> prompt = LLaMA2.get_prompt("Hi do you know Pytorch?")
            >>> for response in llama2_wrapper.generate(prompt):
            ...     logger.info(response)

        Args:
            prompt: The prompt to generate text from.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Yields:
            The generated text.
        """
        if self.backend_type is BackendType.LLAMA_CPP:
            inputs = self.model.tokenize(bytes(prompt, "utf-8"))

            generator = self.model.generate(
                inputs,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
            outputs = []
            for token in generator:
                if token == self.model.token_eos():
                    break
                b_text = self.model.detokenize([token])
                text = str(b_text, encoding="utf-8", errors="ignore")
                outputs.append(text)
                yield "".join(outputs)
        else:
            from transformers import TextIteratorStreamer

            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                # num_beams=1,
            )
            generate_kwargs = (
                generate_kwargs if kwargs is None else {**generate_kwargs, **kwargs}
            )
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()

            outputs = []
            for text in streamer:
                outputs.append(text)
                yield "".join(outputs)

    def run(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
    ) -> Iterator[str]:
        """Create a generator of response from a chat message.
        Process message to llama2 prompt with with chat history
        and system_prompt for chatbot.

        Args:
            message: The origianl chat message to generate text from.
            chat_history: Chat history list from chatbot.
            system_prompt: System prompt for chatbot.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Yields:
            The generated text.
        """
        prompt = LLaMA2.get_prompt(message, chat_history, system_prompt)
        return self.generate(
            prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.

        Examples:
            >>> llama2_wrapper = LLaMA2()
            >>> prompt = LLaMA2.get_prompt("Hi do you know Pytorch?")
            >>> logger.info(llama2_wrapper(prompt))

        Args:
            prompt: The prompt to generate text from.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Generated text.
        """
        if self.backend_type is BackendType.LLAMA_CPP:
            output = self.model.__call__(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
            return output["choices"][0]["text"]
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
            output_ids = self.model.generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
            output = self.tokenizer.decode(output_ids[0])
            return output.split("[/INST]")[1].split("</s>")[0]



class BackendType(Enum):
    TRANSFORMERS = 1
    GPTQ = 2
    LLAMA_CPP = 3

    @classmethod
    def get_type(cls, backend_name: str):
        backend_type = None
        backend_name_lower = backend_name.lower()
        if "transformers" in backend_name_lower:
            backend_type = BackendType.TRANSFORMERS
        elif "gptq" in backend_name_lower:
            backend_type = BackendType.GPTQ
        elif "cpp" in backend_name_lower:
            backend_type = BackendType.LLAMA_CPP
        else:
            raise Exception("Unknown backend: " + backend_name)
        return backend_type


class ModelType(Enum):
    LLAMA2 = 1

    __MODEL_MAP__ = {
        LLAMA2: LLaMA2,
    }

    @classmethod
    def get_type(cls, model_name: str):
        model_type = None
        model_type_lower = model_name.lower()
        if "llama2" in model_type_lower:
            model_type = ModelType.LLAMA2
        else:
            raise Exception("Unknown model: " + model_name)
        return model_type
    
    @classmethod
    def get_model(cls, model_name: str) -> Any:
        return ModelType.__MODEL_MAP__[ModelType.get_type(model_name).value]
