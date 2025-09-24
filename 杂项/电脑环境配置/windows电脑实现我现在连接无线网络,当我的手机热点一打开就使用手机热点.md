windows电脑如何实现我现在连接时的bdas无线网络,当我的手机热点ttk一打开就使用手机热点ttk呢
我帮你梳理一下这个需求：

👉 你现在在 Windows 电脑上，平时用 **bdas 无线网络**上网，但如果你打开手机热点（名字叫 **ttk**），希望电脑能 **自动切换到手机热点 ttk** 使用。

这其实是 **Wi-Fi 自动优先级设置**的问题。

---

## 方法 1：直接用 Windows 图形界面设置 Wi-Fi 优先级

Windows 10/11 里默认没有明显的 Wi-Fi 优先级设置按钮，但是可以通过 **记住网络 + 删除重连顺序**来实现：

1. 打开 **设置 → 网络和 Internet → Wi-Fi → 管理已知网络**。
    
2. 找到 **ttk**，点进去 → 点击“自动连接”选项，确保开启。
    
3. 把 **bdas** 的“自动连接”取消勾选。
    
    - 这样一旦 **ttk** 热点可用，系统会优先连 ttk；
        
    - ttk 不在范围时，才会去连 bdas。
        

---

## 方法 2：命令行设置 Wi-Fi 优先级（更灵活）

Windows 下可以用 `netsh wlan` 命令修改优先级。

1. 打开 PowerShell（管理员模式）。
    
2. 查看已保存的 Wi-Fi 配置：
    
    ```powershell
    netsh wlan show profiles
    ```
    
    假设输出中有：
    
    - `bdas`
        
    - `ttk`
        
3. 设置 **ttk 的优先级高于 bdas**（优先级数值越小，优先级越高）：
    
    ```powershell
    netsh wlan set profileorder name="ttk" interface="Wi-Fi" priority=1
    netsh wlan set profileorder name="bdas" interface="Wi-Fi" priority=2
    ```
    
    （这里 `"Wi-Fi"` 是网卡名字，如果叫别的，可以先用 `netsh wlan show interfaces` 查看。）
    

---

## 方法 3：第三方工具（比如 Wifinian）

如果你觉得命令行麻烦，可以用开源小工具 [Wifinian](https://github.com/emoacht/Wifinian)，它能可视化地调整 Wi-Fi 连接优先级，并支持快速切换。

---

⚡ 建议：

- 如果你只想“热点一开就自动连”，用 **方法 1 最简单**：给 ttk 勾选自动连接，把 bdas 取消自动连接即可。
    
- 如果还想要 **更精细的控制顺序**（例如 ttk > bdas > 其它 Wi-Fi），用 **方法 2 或 Wifinian**。
    

---

要不要我帮你写一个 **PowerShell 自动切换脚本**，比如每次启动电脑时自动检查 ttk 热点，如果存在就直接切过去？
```cmd
netsh wlan show profiles

```

![](https://tk-pichost-1325224430.cos.ap-chengdu.myqcloud.com/blog/20250924094020418.png?imageSlim)![](https://tk-pichost-1325224430.cos.ap-chengdu.myqcloud.com/blog/20250924094155132.png?imageSlim)
你遇到的报错原因我看出来了 👉 这里的 `interface` 参数，不是填网卡的全称，也不是随便写，而是要填 **接口的名字**，就是 `netsh wlan show interfaces` 输出里 `名称` 那一行的值。

你这里显示是：

```
名称 : WLAN
说明 : Realtek 8852CE WiFi 6E PCI-E NIC
```

所以正确的命令应该是用 **WLAN**，而不是 Realtek 那一长串。  
同时注意，参数里不要多余的空格。

---

### 正确写法示例：

```powershell
netsh wlan set profileorder name="ttk" interface="WLAN" priority=1
netsh wlan set profileorder name="bdas" interface="WLAN" priority=2
```

这样就能设置 **ttk 热点优先级更高**，当 ttk 可用时，系统会优先连 ttk；如果 ttk 不可用，才会自动回到 bdas。

---

⚡ 小提醒：

- `名称`（WLAN）= 接口名，要填这个。
    
- `配置文件`（ttk、bdas）= Wi-Fi SSID 配置文件名，要填这个。
    
- `说明`（Realtek 8852CE WiFi 6E PCI-E NIC）只是网卡型号，不能用。
    

---

好嘞！下面给你一键脚本（PowerShell），自动做到：

- 先检查本机是否已保存 `ttk` 与 `bdas` 的 Wi-Fi 配置；
    
- 扫描附近热点：若发现 `ttk`，立刻切到 `ttk`；否则连接 `bdas`；
    
- 同时把优先级设置为 `ttk` > `bdas`（同一接口上）。

> **用法**：把下面内容保存为 `Switch-WiFi.ps1`，右键“使用 PowerShell 运行”（或在 PowerShell 里 `.\Switch-WiFi.ps1`）。无需管理员权限。

```powershell
<# ==================== Switch-WiFi.ps1 ==================== #>
# 你可以按需修改这 3 个变量
$InterfaceName = "WLAN"   # 用 netsh wlan show interfaces 看到的 “名称”
$PrimarySSID   = "ttk"    # 手机热点
$FallbackSSID  = "bdas"   # 默认校园/公司网

function HasProfile([string]$name) {
    (netsh wlan show profiles) -match ("所有用户配置文件|User profiles") -or $true | Out-Null
    $profiles = netsh wlan show profiles
    return ($profiles -match ("(?:所有用户配置文件|Profile)\s*:\s*$([regex]::Escape($name))\b"))
}

function CurrentSSID() {
    $out = netsh wlan show interfaces
    foreach ($line in $out) {
        if ($line -match "^\s*SSID\s*:\s*(.+)$") { return $Matches[1].Trim() }
        if ($line -match "^\s*SSID\s*:\s*(.+)$") { return $Matches[1].Trim() }
        if ($line -match "^\s*SSID\s*[:：]\s*(.+)$") { return $Matches[1].Trim() }
    }
    return ""
}

function IsInRange([string]$ssid) {
    $nets = netsh wlan show networks interface="$InterfaceName" mode=bssid 2>$null
    if (-not $nets) { return $false }
    return ($nets -match ("SSID\s+\d+\s*[:：]\s*$([regex]::Escape($ssid))\b"))
}

function EnsurePriority($primary, $fallback) {
    try {
        # 设置优先级：priority 数字越小优先级越高
        netsh wlan set profileorder name="$primary"  interface="$InterfaceName" priority=1 | Out-Null
        netsh wlan set profileorder name="$fallback" interface="$InterfaceName" priority=2 | Out-Null
    } catch { }
}

Write-Host ">>> 目标接口：$InterfaceName"
Write-Host ">>> 主 SSID：$PrimarySSID"
Write-Host ">>> 备 SSID：$FallbackSSID"
Write-Host ">>> 正在检查已保存的 Wi-Fi 配置..."

$hasPrimary  = HasProfile $PrimarySSID
$hasFallback = HasProfile $FallbackSSID

if (-not $hasPrimary) {
    Write-Host "× 未发现已保存的配置：$PrimarySSID"
    Write-Host "  请先手动连接一次 $PrimarySSID（输入密码让系统保存配置），再运行本脚本。"
    exit 1
}
if (-not $hasFallback) {
    Write-Host "！未发现已保存的配置：$FallbackSSID（非必需，但建议保存作为回退）"
}

# 优先级调整
EnsurePriority $PrimarySSID $FallbackSSID

# 当前连接情况
$current = CurrentSSID
if ($current) { Write-Host ">>> 当前连接：$current" } else { Write-Host ">>> 当前未连接 Wi-Fi" }

# 扫描判断
Write-Host ">>> 扫描附近网络..."
$primaryInRange = IsInRange $PrimarySSID

if ($primaryInRange) {
    if ($current -eq $PrimarySSID) {
        Write-Host "✓ 已连接到 $PrimarySSID，无需切换。"
    } else {
        Write-Host "⇒ 发现 $PrimarySSID，正在切换..."
        $rc = netsh wlan connect name="$PrimarySSID" interface="$InterfaceName"
        if ($LASTEXITCODE -eq 0 -or ($rc -join "`n") -match "已成功|completed") {
            Write-Host "✓ 已连接到 $PrimarySSID"
            exit 0
        } else {
            Write-Host "× 切换到 $PrimarySSID 失败，输出："
            Write-Host ($rc -join "`n")
        }
    }
} else {
    Write-Host "！未发现 $PrimarySSID（可能未开启手机热点或不在范围）"
    if ($hasFallback) {
        if ($current -eq $FallbackSSID) {
            Write-Host "✓ 已连接到 $FallbackSSID。"
        } else {
            Write-Host "⇒ 尝试连接到回退网络 $FallbackSSID ..."
            $rc = netsh wlan connect name="$FallbackSSID" interface="$InterfaceName"
            if ($LASTEXITCODE -eq 0 -or ($rc -join "`n") -match "已成功|completed") {
                Write-Host "✓ 已连接到 $FallbackSSID"
                exit 0
            } else {
                Write-Host "× 连接 $FallbackSSID 失败，输出："
                Write-Host ($rc -join "`n")
                exit 2
            }
        }
    } else {
        Write-Host "× 未保存回退网络 $FallbackSSID，且未发现 $PrimarySSID。"
        exit 3
    }
}
<# ================== end of file ================== #>
```

### 运行前小检查

1. 先用你的电脑分别连接一次 `ttk` 和 `bdas`，让系统保存配置（否则脚本没法无密码连接）。
    
2. 确认接口名就是 `WLAN`：
    
    ```powershell
    netsh wlan show interfaces
    ```
    
    若显示的“名称”不是 `WLAN`，把脚本开头的 `$InterfaceName` 改成你的名称即可。
    
### 想要开机或登录自动执行？
可以添加计划任务（无需管理员）让它在登录时自动跑一次：
```powershell
$script = "$env:USERPROFILE\Switch-WiFi.ps1"    # 改成你的脚本路径
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "SwitchWiFiToTTK" -Action $action -Trigger $trigger
```

---

---

##  方案 2：持续监听并自动切换（每 10 秒检查一次）后台运行版（开机自启，不弹出窗口）

如果你想要后台一直跑，发现手机热点 `ttk` 出现就切换过去，可以用这个 PowerShell 版本：
## 🔹 脚本：`AutoSwitch-WiFi.ps1`

保存到： `C:\Users\tk\AutoSwitch-WiFi.ps1`

```powershell
# AutoSwitch-WiFi.ps1
$InterfaceName = "WLAN"   # 接口名称，来自 netsh wlan show interfaces
$PrimarySSID   = "ttk"    # 手机热点
$FallbackSSID  = "bdas"   # 默认网络

function CurrentSSID() {
    $out = netsh wlan show interfaces
    foreach ($line in $out) {
        if ($line -match "^\s*SSID\s*[:：]\s*(.+)$") { return $Matches[1].Trim() }
    }
    return ""
}

function IsInRange([string]$ssid) {
    $nets = netsh wlan show networks interface="$InterfaceName" mode=bssid 2>$null
    if (-not $nets) { return $false }
    return ($nets -match ("SSID\s+\d+\s*[:：]\s*$([regex]::Escape($ssid))\b"))
}

while ($true) {
    $current = CurrentSSID
    if (IsInRange $PrimarySSID) {
        if ($current -ne $PrimarySSID) {
            netsh wlan connect name="$PrimarySSID" interface="$InterfaceName" | Out-Null
        }
    } elseif ($current -ne $FallbackSSID -and $FallbackSSID -ne "") {
        netsh wlan connect name="$FallbackSSID" interface="$InterfaceName" | Out-Null
    }
    Start-Sleep -Seconds 10   # 每 10 秒检查一次
}
```

> ⚠️ 说明：脚本本身没有 `Write-Host` 输出，后台跑时不会弹窗。

---

## 🔹 新建任务计划，让脚本开机后台运行

1. 打开 **任务计划程序**（`Win + R` → 输入 `taskschd.msc`）。
    
2. 选择 **创建任务**（不是“基本任务”）。
    
3. **常规**：
    
    - 名称填：`AutoSwitchWiFi`
        
    - 勾选 **以最高权限运行**
        
    - 勾选 **仅当用户登录时运行**（如果想一直后台跑，可以选“无论是否登录”）。
        
4. **触发器**：
    
    - 新建 → **登录时**。
        
5. **操作**：
    
    - 新建 → 程序/脚本：
        
        ```
        powershell.exe
        ```
        
    - 参数：
        
        ```
        -WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\Users\tk\AutoSwitch-WiFi.ps1"
        ```
        
    - 这样它会后台启动，不会弹 PowerShell 窗口。
        
6. 保存后，重启或注销登录，脚本就会自动在后台运行。
    

---

✨ 效果：

- 每次开机后，系统后台有个 PowerShell 脚本跑着；
    
- 它每隔 10 秒检查一次 Wi-Fi；
    
- 一旦发现 `ttk`，自动切换过去；
    
- 不在范围时，自动回到 `bdas`。
    
---

要不要我帮你再写一个 **停止脚本的小工具（比如 Stop-SwitchWiFi.ps1）**，方便你临时关闭这个后台任务？