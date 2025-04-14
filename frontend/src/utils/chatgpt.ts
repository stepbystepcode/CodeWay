import { useMutation, useQueryClient } from '@tanstack/react-query';

// 后端API请求类型
export type BackendApiRequest = {
  query: string;
  session_id?: string;
  url?: string;
  mode?: string;
};

// 后端API响应类型
export type BackendApiResponse = {
  answer: string;
  session_id: string;
};

// 为了兼容现有代码，保留OpenAI风格的类型定义
export type ChatCompletionRequest = {
  mode: string;
  messages: {
    role: 'system' | 'user' | 'assistant';
    content: string;
  }[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  session_id?: string; // 添加session_id字段用于传递给后端
  url?: string; // 添加url字段
};

export type ChatCompletionResponse = {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    message: {
      role: 'assistant';
      content: string;
    };
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

export type ChatCompletionChunk = {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta: {
      content?: string;
      role?: string;
    };
    finish_reason: string | null;
  }[];
};
// API function to send streaming requests to ChatGPT
export const streamChatGptRequest = async (
  requestData: ChatCompletionRequest,
  onChunk: (chunk: ChatCompletionChunk) => void,
  onError: (error: Error) => void,
  onComplete: () => void
): Promise<void> => {
  try {
    // 从ChatGPT风格的请求中提取最后一条用户消息作为查询
    const lastUserMessage = [...requestData.messages].reverse().find(msg => msg.role === 'user');
    
    if (!lastUserMessage) {
      throw new Error('No user message found in the request');
    }

    // 构建后端API请求
    const backendRequest: BackendApiRequest = {
      query: lastUserMessage.content,
      url: requestData.url,
      mode: requestData.mode || 'local' // 默认使用本地模型
    };

    const response = await fetch(`${import.meta.env.VITE_API_URL}/api/ask-github-docs`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(backendRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP error! status: ${response.status}`);
    }

    // 处理非流式响应（可能是因为服务器不支持流式响应）
    try {
      const data = await response.json();
      
      // 如果服务器返回的是JSON对象，则处理它
      // 检查是否有response或answer字段
      const content = data.response || data.answer || JSON.stringify(data);
      
      // 创建一个模拟块
      const mockChunk: ChatCompletionChunk = {
        id: `chunk-${Date.now()}`,
        object: 'chat.completion.chunk',
        created: Date.now(),
        model: requestData.mode || 'local',
        choices: [
          {
            index: 0,
            delta: {
              content: content
            },
            finish_reason: 'stop'
          }
        ]
      };
      
      // 发送内容并完成
      onChunk(mockChunk);
      onComplete();
      return;
    } catch (e) {
      // 如果不是JSON，则尝试流式处理
      // 继续处理流式响应
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    
    let responseText = '';
    let done = false;
    
    // 直接读取流
    while (!done) {
      const { value, done: doneReading } = await reader.read();
      done = doneReading;

      if (done) {
        // 尝试将完整响应解析为JSON
        try {
          const jsonResponse = JSON.parse(responseText);
          const finalContent = jsonResponse.response || jsonResponse.answer || responseText;
          
          // 发送最终的内容
          const finalChunk: ChatCompletionChunk = {
            id: `chunk-final-${Date.now()}`,
            object: 'chat.completion.chunk',
            created: Date.now(),
            model: requestData.mode || 'local',
            choices: [
              {
                index: 0,
                delta: {
                  content: finalContent
                },
                finish_reason: 'stop'
              }
            ]
          };
          
          onChunk(finalChunk);
        } catch (e) {
          // 如果不是JSON，就使用原始文本
        }
        
        onComplete();
        break;
      }

      // 解码这个块
      const chunk = decoder.decode(value, { stream: true });
      responseText += chunk;
      
      // 为了兼容现有前端代码，将文本块转换为ChatGPT风格的块
      const mockChunk: ChatCompletionChunk = {
        id: `chunk-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
        object: 'chat.completion.chunk',
        created: Date.now(),
        model: requestData.mode || 'local',
        choices: [
          {
            index: 0,
            delta: {
              content: chunk
            },
            finish_reason: null
          }
        ]
      };
      
      // 将这个模拟块传递给回调
      onChunk(mockChunk);
    }
  } catch (error) {
    onError(error instanceof Error ? error : new Error(String(error)));
  }
};

// Custom hook for streaming ChatGPT responses
export const useStreamChatGptMutation = () => {
  const queryClient = useQueryClient();
  
  return {
    streamChatCompletion: (
      requestData: ChatCompletionRequest,
      callbacks: {
        onChunk: (chunk: ChatCompletionChunk) => void;
        onError: (error: Error) => void;
        onComplete: () => void;
      }
    ) => {
      return streamChatGptRequest(
        requestData,
        callbacks.onChunk,
        callbacks.onError,
        () => {
          callbacks.onComplete();
          queryClient.invalidateQueries({ queryKey: ['chatHistory'] });
        }
      );
    }
  };
};

// Utility function to convert our app message format to OpenAI API format
export const convertMessagesToApiFormat = (
  messages: Array<{ role: 'user' | 'assistant'; content: string }>,
  mode: string
): ChatCompletionRequest => {
  return {
    mode,
    messages: messages.map(msg => ({
      role: msg.role,
      content: msg.content,
    })),
    temperature: 0.7,
  };
};
