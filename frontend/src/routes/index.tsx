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
  const [model, setModel] = useState<string>('gpt-4o');
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
      timestamp: new Date()
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
          {activeMessages.map(message => (
            <div key={message.id} className="mb-4">
              {message.role === 'user' ? (
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
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mr-auto max-w-[80%]">
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
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
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