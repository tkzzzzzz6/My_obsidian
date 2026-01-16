

作为一名计算机专业的学生，看到自己的编程投入转化为实实在在的 XP（经验值）是一件非常有成就感的事情。**Code::Stats** 是一个免费的程序员统计服务，而 **codestats.nvim** 正是将这一功能完美集成到 Neovim 中的利器,需要说明的是官方暂时没有提供neovim的插件,本插件是github开源大佬提供的。


## 2. 前置准备

在安装之前，请确保你的系统中已具备以下条件：
1.  **Neovim** 及其基础环境。
2.  **Curl**：用于向服务器发送统计数据。
3.  **Code::Stats 账户**：你需要前往官网注册，并在机器页面（Machine Page）获取你的 **API Key**。
https://codestats.net/my/machines
## 3. 插件安装与配置

推荐使用 **Lazy.nvim** 插件管理器进行安装。为了保持配置整洁，建议在你的配置目录（如 `~/.config/nvim/lua/plugins/`）下新建一个 `codestats.lua` 文件。
windows环境配置目录是`用户名/.config/nvim/lua/plugins/`(如果配置过lazyvim的话),没有的话就手动创建用户名/.config/这个文件夹,并克隆lazyvim官方仓库即可
https://github.com/LazyVim/LazyVim

### 核心配置代码
在用户名/.config/nvim/lua/plugins/创建一个文件名为codestatus.lua的文件,将下面的内容中需要配置的信息填充好,直接复制到codestatus.lua里
**特别注意**：在结构化配置中，文件必须以 **`return`** 关键字开头 [根据对话历史]。

```lua
return {
  'liljaylj/codestats.nvim',
  dependencies = { 'nvim-lua/plenary.nvim' }, -- 必需依赖，提供异步处理能力
  event = { 'TextChanged', 'InsertEnter' },   -- 懒加载：仅在输入文字或进入插入模式时启动
  cmd = { 'CodeStatsXpSend', 'CodeStatsProfileUpdate' }, -- 命令触发加载
  config = function()
    require('codestats').setup {
      username = '<你的用户名>',      -- 用于获取个人资料数据(必须改)
      base_url = 'https://codestats.net', 
      api_key = '<你的 API key>',    -- 你的个人 API 密钥(必须改)
      send_on_exit = true,           -- 退出 nvim 时自动发送 XP
      send_on_timer = true,          -- 开启定时发送功能
      timer_interval = 60000,        -- 建议设置为 60000ms (1分钟)，防止对服务器造成压力
      curl_timeout = 5,              -- 请求超时时间
    }
  end,
}
```

## 4. 进阶：集成到状态栏

如果你想实时看到自己的 XP 或等级，可以将其集成到状态栏中。以常用的 **Lualine** 为例：

```lua
local xp = function()
  -- 获取当前缓冲区对应语言的 XP
  return require('codestats').get_xp(0)
end

require('lualine').setup {
  sections = {
    lualine_x = {
      'filetype',
      { xp, fmt = function(s) return s and (s ~= '0' or nil) and s .. 'xp' end },
    },
  },
}
```

## 5. 常用交互命令

安装完成后，你可以通过以下命令手动管理数据：
*   **`:CodeStatsXpSend`**：立即手动发送当前的 XP 统计。
*   **`:CodeStatsProfileUpdate`**：手动从服务器拉取最新的个人资料数据。

## 6. 常见故障排除

### 1. 报错 `Failed to load ...: return expected`
**原因**：在 `lua/plugins/` 下的文件没有使用 `return { ... }` 结构。
**解决**：确保你的插件配置文件以 `return` 开头，将配置表传递给 Lazy.nvim。

### 2. 插件克隆失败 (`Connection was reset`)
**原因**：通常是网络环境导致无法正常访问 GitHub,刚开始有几次配置都clone不下来,换节点就好了。
**解决**：
*   在 Neovim 中输入 **`:Lazy`**，选中插件并按 **`R`** 键重试。
*   检查你的终端是否配置了正确的网络代理。

### 3. `plenary.nvim` 依赖问题
`codestats.nvim` 依赖 `plenary.nvim` 来处理异步任务（如 `plenary.job`）。只要在 `dependencies` 中声明，Lazy.nvim 会自动为你安装，无需手动干预。

---

## 结语

现在就打开你的 Neovim，开始累积你的编程经验值吧！