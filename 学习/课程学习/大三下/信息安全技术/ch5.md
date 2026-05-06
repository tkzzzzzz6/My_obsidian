消息摘要

数字签名

强弱抗冲突

生日问题
生日悖论
生日攻击

MD5 message digest algorithm 5:消息摘要算法
SHA-1
SHA-256
SHA-512
SHA-3
sm3:https://www.oscca.gov.cn/sca/xxgk/2010-12/17/1002389/files/302a3ada057c4a73830536d03e683110.pdf

奇偶校验
CRC循环冗余校验

bittorrent:

big/little endian

哈希加盐
服务器数据库存储用户密码

彩虹表

哈希以后理论上不可复原

密码存储和消息认证

消息认证技术

加密技术:需要可逆

认证码技术:不需要可逆

cyclinc redundancy check 循环冗余校验 32 位,容易产生碰撞

mac:message authentication code 消息认证码

Hmac:hash-based message authentication code 基于哈希的消息认证码

异或加

$$
Hmac(K, m) = H((K' \oplus opad) \| H((K' \oplus ipad) \| m))
$$

数字签名

计算机病毒防治,病毒文件 md5 或者 sha 码,拉黑对应码的文件达到治理的效果

非对称加密技术 RSA

数字签名:私钥加密
缺点:非对称加密计算量大,文件大小超过某个值,效率太差

解决方法:先对文件进行哈希,得到消息摘要,再对消息摘要进行数字签名

不能使用 md5 或者 sha1 进行数字签名,因为它们的安全性已经被破解了,容易产生碰撞,攻击者可以伪造一个文件,使得它的 md5 或者 sha1 与原文件相同,从而欺骗验证系统