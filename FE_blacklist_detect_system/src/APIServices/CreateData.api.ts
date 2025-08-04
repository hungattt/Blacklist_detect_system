import customAxios from "../utils/customAxios";

type RequestOptions = {
  headers: {
    "Content-Type": string;
  };
};

type postDataCreateDataService = {
  avata: File | null | undefined;
  name: string;
  full_name: string;
  year_of_birth: number;
  hometown: string;
  id_number: string;
  violation: string;
  data: File | File[] | null | undefined;
};

export const postData = async ({
  data,
  options,
}: {
  data: postDataCreateDataService;
  options: RequestOptions;
}) => {
  try {
    const response = await customAxios.post("create_data/", data, options);
    return response;
  } catch (error) {
    console.error("postDataCreateDataError: ", error);
  }
};
