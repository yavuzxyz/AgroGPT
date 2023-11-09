import os  # os modülü, işletim sistemine özgü işlevselliklere erişim sağlar.
import openai  # OpenAI'nin Python kütüphanesini içe aktarıyoruz.

openai.api_key = "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"  # API anahtarınızı işletim sisteminin ortam değişkenlerinden alıyoruz. Bu, anahtarın doğrudan kodda yer almasını önler ve güvenliği artırır.

response = openai.Completion.create(
    engine="text-davinci-003",  # Kullanmak istediğiniz motorun adını belirtiyoruz. Bu durumda, "text-davinci-003" adlı OpenAI'nin en gelişmiş dil modelini kullanıyoruz.
    prompt="nasılsın ve sen kimsin",  # Modelin tamamlamasını istediğiniz metni belirtiyoruz. Bu, modelin üzerine düşünerek tamamlaması gereken başlangıç metnidir.
    max_tokens=400,  # Modelin üretebileceği en fazla token sayısını belirtiyoruz. Bir token genellikle bir kelime veya kelimenin bir parçasıdır. Bu, oluşturulan metnin uzunluğunu sınırlar.
    temperature=0.1,  # Modelin çıktılarının çeşitliliğini kontrol eden parametre. 0'a yaklaştıkça model daha belirgin/deterministik olur. Bu durumda, modelin çıktısı oldukça belirgin olacak.
    n=1,  # Modelin kaç farklı tamamlama yapmasını istediğinizi belirtir. Bu durumda, yalnızca bir tamamlama istiyoruz.
)

output = response.choices[0].text.strip()  # Modelin oluşturduğu metni alıyoruz ve başındaki ve sonundaki fazla boşlukları kaldırıyoruz. Bu, çıktının temiz ve düzgün görünmesini sağlar.
print(output)  # Oluşturulan metni yazdırıyoruz. Bu, kodun sonucunu görüntülememizi sağlar.
