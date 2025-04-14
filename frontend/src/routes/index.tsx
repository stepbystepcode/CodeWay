import { createFileRoute } from '@tanstack/react-router'
import { useState, useMemo, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select"
import { Send, DownloadCloud, Plus, Menu, Search, User, LogOut, Link } from "lucide-react"
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism'
import 'katex/dist/katex.min.css'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { useStreamChatGptMutation, convertMessagesToApiFormat } from '@/utils/chatgpt'
import { useNavigate } from '@tanstack/react-router'
import UrlManager from '@/components/ui/url-manager'

// Create a query client
const queryClient = new QueryClient()

// Generate unique IDs for React keys
let idCounter = 0;
const generateUniqueId = (prefix = '') => {
  const timestamp = Date.now();
  const uniqueId = `${prefix}${timestamp}-${idCounter++}`;
  return uniqueId;
};

// 加载动画组件
const LoadingSpinner = ({className}: {className?: string}) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`animate-spin ${className || ''}`}
    >
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
};

// 思考中提示组件
const ThinkingIndicator = () => {
  const [text, setText] = useState("Thinking");
  
  useEffect(() => {
    const interval = setInterval(() => {
      setText(prev => {
        if (prev === "Thinking") return "Thinking.";
        if (prev === "Thinking.") return "Thinking..";
        if (prev === "Thinking..") return "Thinking...";
        return "Thinking";
      });
    }, 500);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="flex items-center justify-center gap-2 py-6 text-gray-500 italic">
      <LoadingSpinner className="h-5 w-5" />
      <span className="text-base">{text}</span>
    </div>
  );
};

export const Route = createFileRoute('/')({
  component: () => (
    <QueryClientProvider client={queryClient}>
      <ReactQueryDevtools initialIsOpen={false} />
      <ChatGPT />
    </QueryClientProvider>
  ),
})

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  model?: string; // 添加模型信息字段
}

type Chat = {
  id: string;
  title: string;
  messages: Message[];
  timestamp: Date;
}

function ChatGPT() {
  const navigate = useNavigate();
  const [chats, setChats] = useState<Chat[]>([
    {
      id: generateUniqueId('chat-'),
      title: 'New Chat',
      messages: [],
      timestamp: new Date()
    }
  ]);

  const [activeChat, setActiveChat] = useState<string>(chats[0].id);
  const [inputValue, setInputValue] = useState<string>('');
  const [model, setModel] = useState<string>('local');
  const [filterValue, setFilterValue] = useState<string>('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [url, setUrl] = useState<string>('');
  
  // Check authentication status on component mount
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    setIsAuthenticated(!!token);
  }, []);

  // Logout function
  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    setIsAuthenticated(false);
  };
  
  // Use the ChatGPT mutation hooks
  const { streamChatCompletion } = useStreamChatGptMutation();
  
  // Reference to the current streaming message
  const streamingMessageRef = useRef<Message | null>(null);
  
  const activeMessages = chats.find(chat => chat.id === activeChat)?.messages || [];
  
  const filteredChats = useMemo(() => {
    if (!filterValue.trim()) return chats;
    return chats.filter(chat => 
      chat.title.toLowerCase().includes(filterValue.toLowerCase())
    );
  }, [chats, filterValue]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isStreaming) return;
    
    // Check if user is logged in before sending message
    if (!isAuthenticated) {
      // Redirect to login page
      navigate({ to: '/auth', search: { mode: 'login' } });
      return;
    }
    
    const newMessage: Message = {
      id: generateUniqueId('msg-'),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };
    
    // Update current chat with new message
    const updatedChats = chats.map(chat => {
      if (chat.id === activeChat) {
        // If this is the first message, update the chat title
        const updatedTitle = chat.messages.length === 0 ? 
          inputValue.substring(0, 20) + (inputValue.length > 20 ? '...' : '') : 
          chat.title;
        
        return {
          ...chat,
          title: updatedTitle,
          messages: [...chat.messages, newMessage]
        };
      }
      return chat;
    });
    
    setChats(updatedChats);
    setInputValue('');
    
    // Get the current chat after update
    const currentChat = updatedChats.find(chat => chat.id === activeChat);
    if (!currentChat) return;
    
    // Format request in OpenAI API format
    const messages = [
      ...activeMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      })),
      { role: 'user' as const, content: inputValue }
    ];

    const requestData = convertMessagesToApiFormat(messages, model);
    
    // Add URL to the request
    if (url) {
      requestData.url = url;
    }
    
    // Create an initial streaming response message
    const streamingMessage: Message = {
      id: generateUniqueId('msg-'),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      model: model // 存储生成消息时使用的模型
    };
    
    // Store the reference to the current streaming message
    streamingMessageRef.current = streamingMessage;
    
    // Add the initial empty assistant message
    setChats(prevChats => prevChats.map(chat => {
      if (chat.id === activeChat) {
        return {
          ...chat,
          messages: [...chat.messages, streamingMessage]
        };
      }
      return chat;
    }));
    
    // Set streaming state to true
    setIsStreaming(true);
    
    // Send request to ChatGPT API with streaming
    try {
      await streamChatCompletion(
        requestData,
        {
          onChunk: (chunk) => {
            const content = chunk.choices[0]?.delta?.content || '';
            
            if (content) {
              // Update the streaming message with new content
              setChats(prevChats => 
                prevChats.map(chat => {
                  if (chat.id === activeChat) {
                    return {
                      ...chat,
                      messages: chat.messages.map(msg => {
                        if (msg.id === streamingMessageRef.current?.id) {
                          return {
                            ...msg,
                            content: msg.content + content
                          };
                        }
                        return msg;
                      })
                    };
                  }
                  return chat;
                })
              );
            }
          },
          onError: (error) => {
            // Update streaming message with error
            setChats(prevChats => 
              prevChats.map(chat => {
                if (chat.id === activeChat) {
                  return {
                    ...chat,
                    messages: chat.messages.map(msg => {
                      if (msg.id === streamingMessageRef.current?.id) {
                        return {
                          ...msg,
                          content: `Error: ${error.message || 'Unknown error'}`
                        };
                      }
                      return msg;
                    })
                  };
                }
                return chat;
              })
            );
            
            // Reset streaming state
            setIsStreaming(false);
            streamingMessageRef.current = null;
          },
          onComplete: () => {
            // Reset streaming state
            setIsStreaming(false);
            streamingMessageRef.current = null;
          }
        }
      );
    } catch (error) {
      console.error('Failed to stream response:', error);
      setIsStreaming(false);
      streamingMessageRef.current = null;
    }
  };

  const createNewChat = () => {
    const newChat: Chat = {
      id: generateUniqueId('chat-'),
      title: 'New Chat',
      messages: [],
      timestamp: new Date()
    };
    
    setChats([newChat, ...chats]);
    setActiveChat(newChat.id);
  };

  const exportChat = () => {
    const chat = chats.find(c => c.id === activeChat);
    if (!chat) return;
    
    const chatContent = chat.messages.map(msg => 
      `${msg.role === 'user' ? 'User' : 'GPT'}: ${msg.content}`
    ).join('\n\n');
    
    const blob = new Blob([chatContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${chat.title}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev);
    console.log("Toggling sidebar:", !sidebarOpen); // Debug log
  };

  return (
    <div className="flex h-full w-full relative">
      {/* Fixed Header - Always visible */}
      <div className="fixed top-0 left-0 z-30 flex items-center h-12 bg-background border-b w-full px-3">
        <Button variant="ghost" size="icon" className="mr-2" onClick={toggleSidebar}>
          <Menu size={18} />
        </Button>
        
        {/* Smooth transition for these buttons when sidebar collapses */}
        <div className={`flex items-center transition-all duration-300 ${sidebarOpen ? 'ml-[10.5rem]' : 'ml-0'}`}>
          <Button variant="ghost" size="icon" onClick={createNewChat} className="mr-2">
            <Plus size={18} />
          </Button>
          <span className="mr-2 font-bold">CodeWay</span>
          <Select
            value={model}
            onValueChange={setModel}
            
          >
            <SelectTrigger className="w-[180px] h-8">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gpt-4o">ChatGPT-4o</SelectItem>
              <SelectItem value="local">Qwen2.5-32B-Instruct</SelectItem>
            </SelectContent>
          </Select>
        </div>
        
        <div className="ml-auto flex gap-2">
          <Button variant="outline" size="icon" onClick={exportChat}>
            <DownloadCloud size={18} />
          </Button>
          {/* Authentication UI */}
          {isAuthenticated ? (
            <div className="flex items-center gap-2">
              <Button variant="outline" size="icon" title="User Profile">
                <User size={18} />
              </Button>
              <Button variant="outline" size="icon" onClick={handleLogout} title="Logout">
                <LogOut size={18} />
              </Button>
            </div>
          ) : (
            <>
              <Button variant="outline" onClick={() => navigate({ to: '/auth', search: { mode: 'login' } })}>
                Login
              </Button>
              <Button variant="outline" onClick={() => navigate({ to: '/auth', search: { mode: 'register' } })}>
                Register
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Left Sidebar - Position absolute */}
      <div 
        className={`fixed left-0 top-12 bottom-0 z-20 transition-all duration-300 border-r bg-background ${
          sidebarOpen ? 'w-[260px]' : 'w-0 overflow-hidden'
        }`}
      >
        {/* Search Input */}
        <div className="sticky top-0 z-20 bg-background w-full border-b p-3">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={filterValue}
              onChange={(e) => setFilterValue(e.target.value)}
              placeholder="Search chats..."
              className="pl-8 h-8 text-sm"
            />
          </div>
        </div>
        
        {/* Chat List */}
        <div className="p-2 overflow-y-auto h-[calc(100%-56px)]">
          {filteredChats.length === 0 ? (
            <div className="text-center py-4 text-sm text-muted-foreground">
              No matching chats
            </div>
          ) : (
            filteredChats.map(chat => (
              <div
                key={chat.id}
                className={`mb-1 cursor-pointer rounded-lg p-2 text-sm text-ellipsis text-nowrap ${
                  chat.id === activeChat ? 'bg-muted' : 'hover:bg-muted/50'
                }`}
                onClick={() => setActiveChat(chat.id)}
              >
                {chat.title}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Chat Area - Full width with padding */}
      <div className={`w-full pt-12 flex flex-col transition-all duration-300 ${
        sidebarOpen ? 'pl-[260px]' : 'pl-0'
      }`}>
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* 消息列表 */}
          {activeMessages.map(message => (
            <div key={message.id} className="mb-4">
              {(message.role as string) === 'user' ? (
                <div className="ml-auto max-w-[80%]">
                  <div className="rounded-lg bg-muted p-3 markdown-content">
                    <div className="prose prose-headings:mt-2 prose-headings:mb-2 prose-headings:font-bold prose-p:my-1 max-w-none">
                      <Markdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                          code(props) {
                            const {children, className, ...rest} = props
                            const match = /language-(\w+)/.exec(className || '')
                            return match ? (
                              <SyntaxHighlighter
                                style={dracula}
                                language={match[1]}
                                customStyle={{ margin: 0 }}
                              >
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            ) : (
                              <code {...rest} className={className}>
                                {children}
                              </code>
                            )
                          }
                        }}
                      >
                        {message.content}
                      </Markdown>
                      
                      {/* 思考中提示已移至消息列表末尾 */}
                      
                      {(message.role as string) === 'assistant' && (
                        <div className="flex items-center mt-3 text-sm gap-2">
                          <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full text-xs">
                            {message.model === 'local' ? 'Qwen2.5-32B' : 'GPT-4o'}
                          </span>
                          <div className="flex-1"></div>
                          <button 
                            className="text-gray-500 hover:text-gray-700 flex items-center gap-0.5"
                            onClick={(e) => {
                              const target = e.currentTarget;
                              const svg = target.querySelector('svg');
                              if (svg) {
                                // 切换填充状态
                                const isFilled = svg.getAttribute('fill') === 'currentColor';
                                svg.setAttribute('fill', isFilled ? 'none' : 'currentColor');
                              }
                            }}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                            </svg>
                          </button>
                          <button 
                            className="text-gray-500 hover:text-gray-700 flex items-center gap-0.5"
                            onClick={() => {
                              // 移除最后一条助手消息
                              // 使用兼容的方法找到最后一个用户消息的索引
                              let lastUserMessageIndex = -1;
                              for (let i = activeMessages.length - 1; i >= 0; i--) {
                                if ((activeMessages[i].role as string) === 'user') {
                                  lastUserMessageIndex = i;
                                  break;
                                }
                              }
                              
                              if (lastUserMessageIndex !== -1) {
                                const lastUserMessage = activeMessages[lastUserMessageIndex];
                                const updatedChats = chats.map(chat => {
                                  if (chat.id === activeChat) {
                                    // 移除最后一个用户消息后的所有消息
                                    const updatedMessages = chat.messages.slice(0, lastUserMessageIndex + 1);
                                    return { ...chat, messages: updatedMessages };
                                  }
                                  return chat;
                                });
                                
                                setChats(updatedChats);
                                
                                // 使用最后一条用户消息重新发送
                                setTimeout(() => {
                                  // 创建一个事件对象，模拟输入和发送消息
                                  setInputValue(lastUserMessage.content);
                                  // 延迟一下再发送，确保状态已经更新
                                  setTimeout(() => handleSendMessage(), 100);
                                }, 100);
                              }
                            }}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="w-full">
                  <div className="rounded-lg p-3 markdown-content">
                    <div className="prose prose-headings:mt-2 prose-headings:mb-2 prose-headings:font-bold prose-p:my-1 max-w-none">
                      <Markdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                          code(props) {
                            const {children, className, ...rest} = props
                            const match = /language-(\w+)/.exec(className || '')
                            return match ? (
                              <SyntaxHighlighter
                                style={dracula}
                                language={match[1]}
                                customStyle={{ margin: 0 }}
                              >
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            ) : (
                              <code {...rest} className={className}>
                                {children}
                              </code>
                            )
                          }
                        }}
                      >
                        {message.content}
                      </Markdown>
                      
                      {/* 思考中提示已移至消息列表末尾 */}
                      
                      {(message.role as string) === 'assistant' && (
                        <div className="flex items-center mt-3 text-sm gap-2">
                          <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full text-xs">
                            {message.model === 'local' ? 'Qwen2.5-32B' : 'GPT-4o'}
                          </span>
                          <div className="flex-1"></div>
                          <button 
                            className="text-gray-500 hover:text-gray-700 flex items-center gap-0.5"
                            onClick={(e) => {
                              const target = e.currentTarget;
                              const svg = target.querySelector('svg');
                              if (svg) {
                                // 切换填充状态
                                const isFilled = svg.getAttribute('fill') === 'currentColor';
                                svg.setAttribute('fill', isFilled ? 'none' : 'currentColor');
                              }
                            }}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                            </svg>
                          </button>
                          <button 
                            className="text-gray-500 hover:text-gray-700 flex items-center gap-0.5"
                            onClick={() => {
                              // 移除最后一条助手消息
                              // 使用兼容的方法找到最后一个用户消息的索引
                              let lastUserMessageIndex = -1;
                              for (let i = activeMessages.length - 1; i >= 0; i--) {
                                if ((activeMessages[i].role as string) === 'user') {
                                  lastUserMessageIndex = i;
                                  break;
                                }
                              }
                              
                              if (lastUserMessageIndex !== -1) {
                                const lastUserMessage = activeMessages[lastUserMessageIndex];
                                const updatedChats = chats.map(chat => {
                                  if (chat.id === activeChat) {
                                    // 移除最后一个用户消息后的所有消息
                                    const updatedMessages = chat.messages.slice(0, lastUserMessageIndex + 1);
                                    return { ...chat, messages: updatedMessages };
                                  }
                                  return chat;
                                });
                                
                                setChats(updatedChats);
                                
                                // 使用最后一条用户消息重新发送
                                setTimeout(() => {
                                  // 创建一个事件对象，模拟输入和发送消息
                                  setInputValue(lastUserMessage.content);
                                  // 延迟一下再发送，确保状态已经更新
                                  setTimeout(() => handleSendMessage(), 100);
                                }, 100);
                              }
                            }}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
          
          {/* 思考中提示 - 直接放在消息列表末尾 */}
          {isStreaming && <ThinkingIndicator />}
        </div>

        {/* Input Area */}
        <div className="border-t p-4">
          <div className="relative">
            <Textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={isStreaming ? "Waiting for AI response..." : "Input message..."}
              className="pr-10 rounded-lg bg-muted resize-none min-h-[60px] max-h-[200px]"
              disabled={isStreaming}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
            />
            <div className="absolute right-2 bottom-2 flex items-center gap-1">
              <UrlManager url={url} onUrlChange={setUrl} />
              <Button 
                size="icon" 
                variant="ghost" 
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isStreaming}
              >
                <Send size={18} />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}