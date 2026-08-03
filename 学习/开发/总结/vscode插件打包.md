1. 安装 vsce 打包工具(只需一次):
       npm install -g @vscode/vsce


这里用acwing-vscode-plugin插件举例

1. 在项目目录打包:
   cd D:\code\acwing-vscode-plugin
   npm run compile
   vsce package

   这会在项目根目录生成一个 .vsix 文件(如 vscode-acwing-1.0.0.vsix)
   ![1779416828524.png](https://tk-pichost-1325224430.cos.ap-chengdu.myqcloud.com/blog/1779416828524.png)

![1779417129655.png](https://tk-pichost-1325224430.cos.ap-chengdu.myqcloud.com/blog/1779417129655.png)

感觉本质上就是一个压缩包

	2. 安装到 VS Code:
	方式一: 命令行
	code --install-extension vscode-acwing-1.0.0.vsix
	
	方式二: VS Code 界面
	- 打开 VS Code
	- 按 Ctrl+Shift+P → 输入 "Extensions: Install from VSIX..."
	- 选择生成的 .vsix 文件
	
	方法三:右键vsix文件,直接安装
		
![1779416962556.png](https://tk-pichost-1325224430.cos.ap-chengdu.myqcloud.com/blog/1779416962556.png)

    4. 安装后重新加载 VS Code 即可使用