function [ class ] = EM( bow, K )
%
% Your goal of this assignment is implementing your own text clustering algo.
%
% Input:
%     bow: data set. Bag of words representation of text document as
%     described in the assignment.
%
%     K: the number of desired topics/clusters. 
%
% Output:
%     class: the assignment of each topic. The
%     assignment should be 1, 2, 3, etc. 
%
% For submission, you need to code your own implementation without using
% any existing libraries

% YOUR IMPLEMENTATION SHOULD START HERE!

   % Initializing empty gamma array:
   
       [num_docs, num_words] = size(bow);
       gamma = ones(num_docs,K);

   % Initializaing random prior probabilities (pi values):
    
       pi_array = rand(1,K);
       pi_array = pi_array/sum(pi_array);
        
   % Initializing mu values:
   
       mu_array = rand(K,num_words);
       mu_array = mu_array./sum(mu_array,2);
       
   % Generating initial comparison arrays to enter the EM loop:
       comp_gamma = -1.*gamma;
       comp_pi = -1.*pi_array;
       comp_mu = -1.*mu_array;
   loop = 0;
   % EM Loop:
   while(~all(all((abs(gamma - comp_gamma) < 1e-30))) && ~all(all(abs(pi_array - comp_pi) < 1e-30)) && ~all(all(abs(mu_array - comp_mu) < 1e-30)))
       % Updating comparison arrays:
           comp_gamma = gamma;
           comp_pi = pi_array;
           comp_mu = mu_array;

       % Expectation (E-Step):

           big_pi = zeros(num_docs,K);
           for c = 1:K
              big_pi(:,c) = prod((mu_array(c,:).^bow),2);
           end
           gamma = pi_array.*(big_pi);
           gamma = gamma./sum(gamma,2);

       % Maximization (M-Step):

           mu_array = zeros(K,num_words);
           for c = 1:K
              mu_numerator = sum(gamma(:,c).*bow);
              mu_denominator = sum(sum((gamma(:,c).*bow),2));
              mu_array(c,:) = (mu_numerator./mu_denominator);
           end
           pi_array = mean(gamma);
       loop = loop + 1;
   end
   
   % Computing hard cluster assignments based on the highest probabilities
   class = zeros(num_docs, 1);
   for i = 1:num_docs
      [~,idx] =  max(gamma(i,:));
      class(i) = idx;
   end
end

