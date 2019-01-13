package min.run;

import java.util.List;

import min.attlda.AttLDA;
import min.attlda.Corpus;
import min.util.URLs;
import org.junit.Test;

public class AttLDATest
{
	@Test
	// Train the Att-LDA model
	public void TestMain() throws Exception
	{
		int T = 20; // The number of latent topics
		double alpha = 0.1; // The prior hyperparameter of documentToTopic 
		double beta = 0.5; // The prior hyperparameter of topicToTerm
		int iters = 1000; // The iteration time
//		 1. Load corpus from disk
		Corpus corpus = new Corpus();
		corpus.load(URLs.APIDescsWeights);
		corpus.saveFiles();
        // 2. Create a Att-LDA sampler
		AttLDA lda = new AttLDA(corpus.getDocuments(), corpus.getWeights(), corpus.getVocabularySize());
        // 3. Training
		lda.gibbs(T, alpha, beta, iters);
        // 4. Save model
        String modelName = "service_attlda";
        
//         5. Calculate the top-k similar terms for a given term
		AttLDA attlda = new AttLDA();
		String word = "search";
		List<String> topWords = attlda.getTopKNeighbors(word, 500);
		System.out.println("The source word is: " + word + "\n");
		for(String s : topWords)
		{
			System.out.println(s);
		}
	}
}