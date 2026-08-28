# 07-2. 두 대 토폴로지와 사전 점검

이 페이지는 07-1. 두 대 연결하기의 상세 내용입니다.

두 대를 연결하는 목적부터 정한다. 모델을 하나의 큰 pool로 만들 것인지, 각 장비에 서버를 하나씩 띄울 것인지에 따라 TP와 DP가 달라진다.

## 선택

- TP: 하나의 요청을 두 노드가 나눠 처리한다.
- DP: 각 노드가 독립 endpoint가 되어 요청을 나눠 받는다.
- PP: layer를 노드 사이에 배치한다.

먼저 NVIDIA의 ConnectX·RoCE·NCCL recipe와 포트 이름을 확인한다. 두 노드의 driver, CUDA, NCCL, hostname, 주소, MTU를 맞춘다.

## 연결 전 체크

```bash
hostname
ip -br addr
ip route
ibdev2netdev || true
nvidia-smi
```

link가 올라온 것과 NCCL collective가 정상인 것은 다르다. network smoke, NCCL test, 실제 model request를 순서대로 실행한다.

## 물리 포트와 Linux 인터페이스를 따로 확인한다

일부 GB10 socket-direct 구성에서는 물리 QSFP 포트 하나가 Linux 인터페이스 여러 개로 보일 수 있다. 인터페이스 이름만 보고 bond를 만들거나 하나를 임의로 버리지 않는다.

```bash
ip -br link
cat /sys/class/net/*/phys_switch_id 2>/dev/null
cat /sys/class/net/*/phys_port_name 2>/dev/null
ibdev2netdev
```

두 rail에 각각 IP를 줄지는 현재 NVIDIA 플레이북과 NCCL recipe를 기준으로 결정한다. 링크가 200G로 협상된 뒤에도 payload가 낮을 수 있으므로 다음을 별도 단계로 기록한다.

```text
link up → IP connectivity → RDMA bandwidth → NCCL collective → model request
```

`ping` 성공이나 `ethtool`의 200G 표시만으로 TP=2를 시작하지 않는다. 상세한 ASUS GX10 현장 사례와 독립 검증 큐는 [노드 세팅 자료 검토](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-node-setup-research-2026-08.md)에 정리했다.
