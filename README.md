# Projeto: Prototipagem de Posicionamento Indoor com BLE

## Sobre nós
Desenvolvido para o Challenge FIAP 2025 por:

Grupo LLM
- Gabriel Marques de Lima Sousa - RM 554889
- Leonardo Matheus Teixeira - RM 556629
- Leonardo Menezes Parpinelli Ribas - RM 557908

## Descrição
Este projeto é um protótipo de sistema de posicionamento indoor utilizando tecnologia BLE (Bluetooth Low Energy). O objetivo é facilitar a localização de motos nos pátios da Mottu, por meio de beacons/tags e antenas baseadas em ESP32.

### Funcionamento do Sistema
- **Antenas ESP32:** Dispositivos ESP32 configurados como antenas escaneiam continuamente o ambiente em busca de beacons/tags BLE conhecidos.
- **Beacons/Tags ESP32:** Outros ESP32 atuam como beacons/tags, podendo ser ativados para emitir um som (buzzer) e acender um LED, auxiliando na localização física da moto.
- **Servidor HTTP:** As antenas enviam os dados de escaneamento (RSSI, identificador do beacon, timestamp, etc.) para um servidor HTTP, que armazena essas informações no Firebase Realtime Database.
- **Cloud Functions no Firebase:**
	- Uma função armazena o histórico de RSSI para cada beacon/tag, permitindo filtragem e cálculo de médias.
	- Outra função calcula a posição estimada do beacon/tag com base nos valores de RSSI recebidos pelas antenas.
- **Dashboard Web:** Uma interface web simples exibe o mapa em tempo real dos beacons/tags e gráficos dos valores de RSSI, facilitando o acompanhamento e análise dos dados.

## Tecnologias Utilizadas
- **ESP32:** Microcontrolador utilizado tanto nas antenas quanto nos beacons/tags.
- **Bluetooth Low Energy (BLE):** Comunicação sem fio de baixo consumo para detecção e localização.
- **Firebase Realtime Database:** Armazenamento dos dados de escaneamento e resultados de posicionamento.
- **Firebase Cloud Functions:** Processamento dos dados, filtragem de RSSI e cálculo de posição.
- **Node.js:** Backend do servidor HTTP para receber e encaminhar os dados das antenas.
- **HTML, CSS, JavaScript:** Desenvolvimento do dashboard web para visualização dos dados.

## Como Executar
### 1. Antenas e Beacons/Tags
- Grave o firmware correspondente nos ESP32 das antenas e dos beacons/tags (código disponível na pasta `iot/`).
- Configure as credenciais de rede Wi-Fi e parâmetros de identificação nos arquivos de configuração (`env.h`).

### 2. Firebase
- Crie um projeto Firebase e configure os arquivos em `firebase/`.
- Faça deploy das Cloud Functions presentes em `firebase/functions/`.
- Certifique-se de que o arquivo `sa.json` contém as credenciais corretas de acesso ao Firebase.

### 3. Servidor HTTP
- Instale as dependências do Node.js na pasta `http-server/`:
	```powershell
	cd http-server
	npm install
	```
- Inicie o servidor:
	```powershell
	node index.js
	```
- Certifique-se de que o arquivo `sa.json` contém as credenciais corretas de acesso ao Firebase.

### 4. Dashboard Web
- Abra o arquivo `dashboard/index.html` em um navegador web para visualizar o mapa e os gráficos em tempo real.

## Resultados Parciais
- O sistema já é capaz de detectar beacons/tags BLE em tempo real e exibir seus sinais RSSI no dashboard.
- As funções de filtragem e cálculo de posição estão implementadas e em testes, permitindo estimativas de localização com base nos dados coletados.
- O acionamento remoto de buzzer e LED nos beacons/tags facilita a localização física dos dispositivos.
- O dashboard apresenta visualização clara dos dados, com gráficos de RSSI e mapa dos dispositivos, porém a experiência e a precisão ainda podem ser aprimoradas.

## Observações
Este projeto é um protótipo e está em constante evolução. Novas funcionalidades e melhorias de precisão estão sendo desenvolvidas.
